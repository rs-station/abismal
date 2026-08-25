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


@pytest.mark.xfail(
    strict=True,
    reason="_tail only clears _monitoring_active on the no-refinement path. With "
           "has_phenix=True it is never cleared, so the Colab poll callback keeps "
           "returning True and the browser interval runs for the life of the tab.",
)
def test_monitoring_stops_after_a_refinement_run(runner_factory, fake_colab, tmp_path):
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
