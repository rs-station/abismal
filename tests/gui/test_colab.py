"""Colab behaviour, tested without Colab.

Colab is the one environment that cannot be automated locally, and it is also where
the failures are least visible: an exception from a click handler goes nowhere, and
widget traits set from a background thread never reach the frontend. Both problems are
worked around in the source, and both workarounds are testable here -- what is left for
a human is only whether the pixels appear.

`_is_colab` is imported *by value* into runner.py, so it must be patched on the runner
module. Patching file_selector._is_colab has no effect.
"""
import ast
import sys
import types
from pathlib import Path

import pytest

import abismal.gui.components.argparse_gui as argparse_gui_module
import abismal.gui.runner as runner_module


@pytest.fixture
def fake_colab(monkeypatch):
    """Install a fake google.colab and make _is_colab report True.

    Returns the list the fake register_callback records into, so a test can pull the
    poll callback out and invoke it directly.
    """
    registered = []

    output_module = types.ModuleType("google.colab.output")
    output_module.register_callback = lambda name, fn: registered.append((name, fn))
    colab = types.ModuleType("google.colab")
    colab.output = output_module
    google = types.ModuleType("google")
    google.colab = colab

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.output", output_module)
    monkeypatch.setattr(runner_module, "_is_colab", lambda: True)
    return registered


# ---------------------------------------------------------------------------
# the poll callback
# ---------------------------------------------------------------------------

def test_a_poll_callback_is_registered_on_colab(runner_factory, fake_colab):
    runner_factory()
    assert len(fake_colab) == 1
    name, _ = fake_colab[0]
    assert name.startswith("abismal_runner_poll_")


def test_no_callback_is_registered_off_colab(runner_factory):
    runner_factory()  # _is_colab is the real one, and google.colab is not importable
    # nothing to assert against directly; the point is that construction did not raise
    # and did not need Colab


def test_the_driving_javascript_carries_the_callback_id(runner_factory, fake_colab):
    runner = runner_factory()
    name, _ = fake_colab[0]

    scripts = [js for _, js in _scripts(runner._colab_poll_widget)]
    assert scripts and name in scripts[0]
    assert "invokeFunction" in scripts[0]


def test_polling_pushes_the_widgets_the_frontend_needs(runner_factory, fake_colab):
    """Colab does not sync traits set from a background thread, so the callback has to
    push each one explicitly. A widget missing here silently stops updating."""
    runner = runner_factory(has_phenix=False)
    _, callback = fake_colab[0]
    pushed = _spy_on_send_state(runner)

    callback()

    assert pushed[runner.log_widget] == [["value"]]
    assert pushed[runner.progress_widget] == [["value", "max", "bar_style"]]
    assert pushed[runner.progress_label] == [["value"]]
    assert pushed[runner.stop_button] == [["disabled"]]
    assert pushed[runner.history_widget] == [["outputs"]]


def test_the_viewer_is_pushed_too_when_there_is_one(runner_factory, fake_colab):
    runner = runner_factory(has_phenix=True)
    _, callback = fake_colab[0]
    pushed = _spy_on_send_state(runner)

    callback()

    assert runner.viewer_widget in pushed
    assert runner._js_widget in pushed


def test_polling_reports_whether_to_keep_going(runner_factory, fake_colab):
    """The browser clears its interval when the callback returns false, so this is the
    only thing that ever stops the polling loop."""
    runner = runner_factory()
    _, callback = fake_colab[0]

    assert callback() is True

    runner._monitoring_active = False
    assert callback() is False


def test_a_broken_widget_does_not_stop_the_poll(runner_factory, fake_colab):
    """One widget failing must not take the whole sync down with it."""
    runner = runner_factory()
    _, callback = fake_colab[0]
    _spy_on_send_state(runner)

    def explode(*a, **k):
        raise RuntimeError("frontend gone")

    runner.log_widget.send_state = explode

    assert callback() is True


def test_monitoring_stops_after_a_refinement_run(runner_factory, fake_colab, tmp_path):
    """_tail clears _monitoring_active on the no-refinement path; the branch with
    refinement has to clear it when its watcher gives up, or the poll callback keeps
    returning True and the browser interval runs for the life of the tab.
    """
    runner = runner_factory(has_phenix=True, out_dir=str(tmp_path))
    _, callback = fake_colab[0]

    runner._monitoring_active = True
    # what _tail does when the process ends, for the has_phenix branch
    runner._post_training_phenix_watcher(max_unchanged=1)

    assert callback() is False


# ---------------------------------------------------------------------------
# _run_on_main_thread -- the reason background updates work on Colab at all
# ---------------------------------------------------------------------------

def test_updates_are_applied_inline_without_a_kernel(runner_factory):
    """This is what makes the whole headless harness work."""
    runner = runner_factory()
    runner._append_log("inline\n")
    assert "inline" in runner.log_widget.value


def test_updates_are_deferred_to_the_kernel_loop_when_there_is_one(
    runner_factory, monkeypatch
):
    """Under a kernel the mutation must be marshalled onto the event loop instead.

    On Colab, a trait set from a background thread never reaches the frontend, which
    is why this indirection exists at all.
    """
    scheduled = []

    class FakeLoop:
        def add_callback(self, fn):
            scheduled.append(fn)

    class FakeKernel:
        io_loop = FakeLoop()

    class FakeShell:
        kernel = FakeKernel()

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: FakeShell()
    monkeypatch.setitem(sys.modules, "IPython", ipython)

    runner = runner_factory()
    runner._append_log("deferred\n")

    assert scheduled, "the update should have been queued, not applied"
    assert "deferred" not in runner.log_widget.value

    for fn in scheduled:
        fn()
    assert "deferred" in runner.log_widget.value


# ---------------------------------------------------------------------------
# structural guards -- cheap, and they catch Colab-only failures at CI time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module", [runner_module, argparse_gui_module], ids=["runner", "argparse_gui"]
)
def test_no_display_calls(module):
    """Every output must be a widget trait assignment.

    display() from a background thread goes nowhere on Colab, and these modules
    currently avoid it only by discipline -- runner.py even imports it without using
    it. This makes the property explicit.
    """
    tree = ast.parse(Path(module.__file__).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "display"
    ]
    assert not calls, [node.lineno for node in calls]


def _spy_on_send_state(runner):
    """Record send_state calls per widget, without a frontend."""
    pushed = {}

    def spy_for(widget):
        def send_state(traits=None, **kwargs):
            pushed.setdefault(widget, []).append(list(traits) if traits else None)

        return send_state

    for widget in _iter(runner.to_widget()):
        widget.send_state = spy_for(widget)
    return pushed


def _scripts(widget):
    import gui_harness as H

    return H.extract_scripts(widget)


def _iter(widget):
    yield widget
    for child in getattr(widget, "children", None) or ():
        yield from _iter(child)


# ---------------------------------------------------------------------------
# where the file pickers open
# ---------------------------------------------------------------------------

def test_content_beats_a_live_proc_cwd_on_colab(monkeypatch):
    """Regression: /content used to be unreachable on Colab.

    The fallback was written as though the launch-directory lookup would fail
    there. It does not. Colab sets JPY_PARENT_PID, so /proc/<pid>/cwd resolves
    -- to the cwd of a supervisor process the user has never seen -- and that
    plausible-looking wrong answer returned before the fallback ever ran. Only
    the ordering makes /content reachable, so only the ordering is tested.
    """
    import os
    from abismal.gui.components import file_selector as fs

    real_isdir = os.path.isdir
    monkeypatch.setattr(fs, "_is_colab", lambda: True)
    monkeypatch.setenv("JPY_PARENT_PID", str(os.getpid()))
    monkeypatch.setattr(
        fs.os.path, "isdir", lambda p: p == "/content" or real_isdir(p)
    )

    assert fs.default_directory() == "/content"


def test_the_launch_directory_still_wins_off_colab(monkeypatch):
    """The Colab branch must not capture the ordinary JupyterLab case."""
    import os
    from abismal.gui.components import file_selector as fs

    real_isdir = os.path.isdir
    monkeypatch.setattr(fs, "_is_colab", lambda: False)
    monkeypatch.setattr(
        fs.os.path, "isdir", lambda p: p == "/content" or real_isdir(p)
    )
    monkeypatch.setattr(fs, "_jupyter_launch_directory", lambda: "/somewhere/else")

    assert fs.default_directory() == "/somewhere/else"


# ---------------------------------------------------------------------------
# the install cell
# ---------------------------------------------------------------------------

def _install_cell_source():
    import json
    import abismal.gui

    nb = json.loads(
        (Path(abismal.gui.__file__).parent / "abismal_gui.ipynb").read_text()
    )
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    installs = [src for src in code if "pip" in src and "install" in src]
    assert len(installs) == 1, f"expected one install cell, found {len(installs)}"
    return installs[0]


def _install_cell_code():
    """The install cell's code with comments removed.

    Matching a guard against raw source lets it match its own explanation -- the
    comment saying "check_call sends output nowhere" reads exactly like the call
    it forbids. That has now happened twice here, so the guards that forbid a
    construct look only at code.
    """
    import io
    import tokenize

    source = _install_cell_source()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return " ".join(
        token.string for token in tokens if token.type != tokenize.COMMENT
    )






def _output_payloads(module):
    """Every `{'output_type': ..., 'data': ..., 'metadata': ...}` literal in a module.

    Returns (data_keys, metadata_source) per payload, read statically -- these are
    built and assigned in branches that a unit test would have to drive a whole run
    to reach.
    """
    tree = ast.parse(Path(module.__file__).read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "output_type" not in keys:
            continue
        entry = dict(zip(keys, node.values))
        data = entry.get("data")
        data_keys = (
            [k.value for k in data.keys if isinstance(k, ast.Constant)]
            if isinstance(data, ast.Dict)
            else []
        )
        meta = entry.get("metadata")
        found.append((data_keys, ast.dump(meta) if meta is not None else ""))
    return found


def test_html_outputs_are_isolated():
    """A full HTML document in an output area must not inherit the page.

    The viewer is a complete document with a global `*` reset, so without the
    isolated flag -- which renders it in an iframe -- its CSS applies to the
    notebook around it. Colab may ignore the flag, which is a separate question;
    this only asserts we always ask.
    """
    checked = 0
    for data_keys, metadata in _output_payloads(runner_module):
        if "text/html" in data_keys:
            checked += 1
            assert "isolated" in metadata, (
                f"text/html payload with metadata {metadata!r} is not isolated"
            )
    # Without this the test passes by finding nothing, which is exactly what would
    # happen if the payload were ever built somewhere the AST walk cannot see.
    assert checked, "no text/html payload found; the guard is not looking anywhere"


def test_the_injected_scripts_are_an_inventory_of_three():
    """Every one of these is a Colab risk, so adding one should be deliberate.

    They are written into an Output widget's `.outputs` as application/javascript,
    and whether Colab's widget manager executes that MIME type at all is unproven
    -- the autoscroll, the poll loop that is the only way background threads reach
    the frontend, and the viewer's cross-frame reload all depend on it. If this
    count changes, the new script needs checking against a real Colab tab rather
    than against JupyterLab.
    """
    scripts = [
        keys for keys, _ in _output_payloads(runner_module)
        if "application/javascript" in keys
    ]
    assert len(scripts) == 3, f"expected 3 injected scripts, found {len(scripts)}"


def test_the_viewer_filters_the_broadcast_by_its_own_id():
    """The parent now broadcasts, so the receiver has to discriminate.

    Without this a second viewer in the same notebook would reload on another
    viewer's epoch.
    """
    from abismal.gui.components import gemmimol

    template = gemmimol.viewer_template
    assert "window.ABISMAL_VIEWER_ID" in template
    assert "msg.viewer_id" in template, "the viewer must check the id it was sent"



def test_the_gui_extra_carries_no_frontend():
    """[gui] is what a notebook needs anywhere; JupyterLab is its own extra.

    Measured in the official Colab image: jupyterlab drags in six further
    packages -- jupyterlab_server, jupyter-lsp, json5, async-lru, jupyter_builder
    -- into a live kernel that already has a frontend and no use for any of them.
    Every package an install need not change is one that cannot break the kernel
    it is changing.
    """
    import tomllib

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not running from a checkout")

    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    assert not any("jupyterlab" in dep for dep in extras["gui"]), extras["gui"]
    assert any("jupyterlab" in dep for dep in extras["lab"]), extras["lab"]


def test_tensorflow_is_a_range_not_a_pin():
    """Regression: `tensorflow==2.18.0` made abismal uninstallable on Colab.

    TF 2.18 caps ml-dtypes below 0.5; Colab's preinstalled jax requires 0.5 or
    newer; and TF 2.18 imports jax unconditionally. Pinning to it meant
    `import tensorflow` raised before any abismal code ran -- and nowhere else,
    because no developer environment here has jax.
    """
    import tomllib

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not running from a checkout")

    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    tf = [d for d in deps if d.startswith("tensorflow>") or d.startswith("tensorflow=")]
    assert tf, deps
    assert "==" not in tf[0], f"an exact pin locks out Colab's stack: {tf[0]}"



def test_colab_gets_a_widget_manager_that_can_render_v8(monkeypatch):
    """Regression: the form rendered as nothing at all on Colab.

    Colab's built-in widget manager is the ipywidgets 7 generation; abismal
    needs 8. Handed v8 widgets it produces no output and no error -- the cell
    just succeeds and the area under it is blank. Colab's opt-in CDN manager
    (html-manager 5.x) is the documented way out, so abismal.gui asks for it
    rather than leaving it to notebook boilerplate.
    """
    import abismal.gui as gui_module

    called = []
    fake_output = types.SimpleNamespace(
        enable_custom_widget_manager=lambda: called.append(True)
    )
    colab = types.ModuleType("google.colab")
    colab.output = fake_output
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.output", fake_output)

    assert gui_module.enable_colab_widget_manager() is True
    assert called, "enable_custom_widget_manager was never called"


def test_the_widget_manager_call_cannot_break_the_import(monkeypatch):
    """Best-effort: a failure here must not take down an otherwise fine import."""
    import abismal.gui as gui_module

    def boom():
        raise RuntimeError("no frontend")

    fake_output = types.SimpleNamespace(enable_custom_widget_manager=boom)
    colab = types.ModuleType("google.colab")
    colab.output = fake_output
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.output", fake_output)

    assert gui_module.enable_colab_widget_manager() is False


# ---------------------------------------------------------------------------
# ipywidgets 7 and 8
# ---------------------------------------------------------------------------

def test_ipywidgets_7_is_supported():
    """Colab ships 7.7.1 and cannot render what 8 emits.

    Its opt-in CDN widget manager is pinned to an ipywidgets-8 alpha
    (html-manager 5.0.0a) that knows ButtonStyleModel but not HTMLStyleModel or
    TextStyleModel, which 8.1.9 attaches to every HTML and Text widget. The
    browser console says

        Cannot find model module @jupyter-widgets/controls@2.0.0, HTMLStyleModel

    and the widget renders as nothing -- no output, no Python error. A form of
    108 Text and 102 HTML widgets is a blank cell. On 7 everything carries
    DescriptionStyleModel or ButtonStyleModel, which Colab has always rendered.
    """
    import tomllib

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not running from a checkout")

    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    spec = [d for d in extras["gui"] if "ipywidgets" in d]
    assert spec, extras["gui"]
    assert ">=8" not in spec[0], (
        f"{spec[0]!r} forces an upgrade Colab's widget manager cannot render"
    )


def test_tooltips_go_on_controls_not_containers():
    """The trait name differs between majors, and the target must not.

    ipywidgets 8 puts `tooltip` on everything; 7 has it on Button alone and
    spells it description_tooltip elsewhere. Passing the wrong one to a 7.x
    widget is not an error -- traitlets warns and drops it -- which is how
    tooltips went missing on dropdowns before.

    They must also land on the control rather than the HBox wrapping it: a
    container has no description, and ipywidgets 8's tooltip view reads
    description.length, so a tooltip on a Box takes the frontend down.
    """
    from abismal.gui import ArgparseGUI
    from abismal.gui.components._compat import set_tooltip

    containers = {"Box", "HBox", "VBox", "GridBox"}
    tipped = []
    stack = [ArgparseGUI().to_widget()]
    while stack:
        widget = stack.pop()
        for name in ("tooltip", "description_tooltip"):
            if widget.has_trait(name) and getattr(widget, name):
                tipped.append(type(widget).__name__)
                break
        stack.extend(getattr(widget, "children", ()) or ())

    assert tipped, "no tooltips were set at all"
    on_containers = [n for n in tipped if n in containers]
    assert not on_containers, f"tooltips sitting on containers: {set(on_containers)}"

    # And the helper picks whichever name this ipywidgets actually uses.
    import ipywidgets as w

    assert set_tooltip(w.Button(description="b"), "x").tooltip == "x"





def test_python_313_is_supported_and_unwheeled_pins_are_marked():
    """Regression: the install failed on Colab for five minutes, then gave up.

    Live Colab is on Python 3.13. `requires-python` ended at <3.13, excluding the
    runtime this GUI exists for, and `scikit-image<0.25` has no cp313 wheel, so
    pip fell back to a 22.7 MB source tarball and died generating its metadata.
    A cap that predates a Python release has to say which Pythons it applies to,
    or it silently becomes "build this from source" on the newest one.

    Two entries, not one marked entry: a failing marker drops a requirement
    rather than relaxing it, and skimage is imported lazily by the torchref peak
    finder -- so losing it would surface halfway through a refinement, not here.
    """
    import tomllib
    from packaging.requirements import Requirement

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not running from a checkout")

    project = tomllib.loads(pyproject.read_text())["project"]
    assert "<3.13" not in project["requires-python"], (
        f"requires-python {project['requires-python']!r} excludes Colab's runtime"
    )

    entries = [d for d in project["dependencies"] if "scikit-image" in d]
    assert entries, project["dependencies"]
    assert any("python_version" in d for d in entries)
    for version in ("3.12", "3.13"):
        applies = [
            d for d in entries
            if Requirement(d).marker is None
            or Requirement(d).marker.evaluate({"python_version": version})
        ]
        assert applies, f"no scikit-image requirement applies on Python {version}"


def test_the_notebook_installs_from_git_not_a_pypi_extra():
    """PyPI's latest abismal publishes only `dev` and `cuda`.

    `pip install abismal[gui]` there warns that the extra does not exist, installs
    no widget stack, and leaves an abismal with no abismal.gui to import. Until a
    release carries the extra, the notebook has to come from git.
    """
    src = _install_cell_source()
    assert "git+" in src, src
