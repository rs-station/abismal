"""The form: which widget each option becomes, what command line it produces, and
what the Run button does.

This is the "widget layout and wiring" half of what used to be checked by looking at a
notebook. All of it runs headlessly -- ipywidgets falls back to a no-op DummyComm with
no kernel attached, so `.value` assignment, `.observe` and `Button.click()` all work.
"""
import argparse

import pytest

import gui_harness as H
from abismal.gui.components.argparse_gui import ArgparseGUI, ArgparseGUIBase


def widget_for(form, dest):
    return {a.dest: w for a, w in form._all_args.items()}[dest]


def inner(widget):
    """The real control inside a _ToggleRow / Text / Dropdown wrapper."""
    return widget.children[-1] if getattr(widget, "children", None) else widget


# ---------------------------------------------------------------------------
# action -> widget
# ---------------------------------------------------------------------------

def test_each_action_kind_becomes_the_right_widget(tiny_form):
    assert type(widget_for(tiny_form, "layers")).__name__ == "Text"
    assert type(widget_for(tiny_form, "activation")).__name__ == "Dropdown"
    assert type(widget_for(tiny_form, "anomalous")).__name__ == "_ToggleRow"
    assert type(widget_for(tiny_form, "no_cache")).__name__ == "_ToggleRow"


def test_choices_become_dropdown_options(tiny_form):
    assert tuple(inner(widget_for(tiny_form, "activation")).options) == ("relu", "swish")


def test_a_default_is_coerced_through_the_actions_type(tiny_form):
    """--lower has choices ('wilson','normal') but default 'Wilson'.

    Without running the default through action.type, traitlets rejects it at
    construction because it is not in options. Several real abismal options are
    exactly this shape.
    """
    assert inner(widget_for(tiny_form, "lower")).value == "wilson"


def test_store_true_starts_off_and_store_false_starts_on(tiny_form):
    assert inner(widget_for(tiny_form, "anomalous")).value is False
    assert inner(widget_for(tiny_form, "no_cache")).value is True


def test_a_text_default_is_a_placeholder_not_a_value(tiny_form):
    """So an untouched field contributes nothing to the command line."""
    control = inner(widget_for(tiny_form, "layers"))
    assert control.value == ""
    assert "5" in str(control.placeholder)


def test_skipped_actions_produce_no_widget():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep", default="x")
    form = H.build_form(parser=parser)

    dests = {a.dest for a in form._all_args}
    assert "debug" in ArgparseGUIBase.skipped_actions
    assert "debug" not in dests
    assert "keep" in dests


# ---------------------------------------------------------------------------
# to_args
# ---------------------------------------------------------------------------

def test_untouched_fields_are_omitted(tiny_form):
    assert "--layers" not in tiny_form.to_args()


def test_a_set_field_is_emitted_as_flag_then_value(tiny_form):
    H.set_control(tiny_form, "layers", "9")
    args = tiny_form.to_args()
    assert args[args.index("--layers") + 1] == "9"


def test_store_true_emits_only_when_on(tiny_form):
    assert "--anomalous" not in tiny_form.to_args()
    H.set_control(tiny_form, "anomalous", True)
    assert "--anomalous" in tiny_form.to_args()


def test_store_false_emits_only_when_off(tiny_form):
    assert "--no-cache" not in tiny_form.to_args()
    H.set_control(tiny_form, "no_cache", False)
    assert "--no-cache" in tiny_form.to_args()


def test_every_argument_is_a_string(tiny_form):
    H.set_control(tiny_form, "layers", "9")
    assert all(isinstance(a, str) for a in tiny_form.to_args())


def test_args_round_trip_through_the_parser(tiny_form, tiny_parser):
    H.set_control(tiny_form, "dmin", "1.8")
    H.set_control(tiny_form, "activation", "swish")
    H.set_control(tiny_form, "anomalous", True)
    H.set_control(tiny_form, "inputs", "a.mtz")

    namespace = tiny_parser.parse_args(tiny_form.to_args())
    assert namespace.dmin == "1.8"
    assert namespace.activation == "swish"
    assert namespace.anomalous is True


# ---------------------------------------------------------------------------
# group container -- the Colab-driven replacement for widgets.Tab
# ---------------------------------------------------------------------------

def group_panels(form):
    return [c for c in form.tab.children if hasattr(c, "layout")]


def test_exactly_one_group_is_visible_at_a_time(tiny_form):
    displays = [p.layout.display for p in tiny_form.children.values()]
    assert displays.count("none") == len(displays) - 1
    assert displays.count("") == 1


def test_clicking_a_group_button_switches_which_one_shows(tiny_form):
    panels = list(tiny_form.children.values())
    if len(panels) < 2:
        pytest.skip("tiny parser produced a single group")
    buttons = [c for c in tiny_form.tab.children[0].children]

    buttons[1].click()

    displays = [p.layout.display for p in panels]
    assert displays[1] == ""
    assert displays.count("") == 1
    assert buttons[1].button_style == "primary"
    assert all(b.button_style == "" for i, b in enumerate(buttons) if i != 1)


# ---------------------------------------------------------------------------
# the Run button
# ---------------------------------------------------------------------------

class FakeRunner:
    """Stands in for AbismalRunner. Records what the form asked it to do."""

    instances = []
    attach_result = None
    # Ordered record across every runner, so a test can assert that the previous one
    # was put down *before* the step that deletes the files it is polling.
    events = []

    def __init__(self, args, out_dir, has_phenix=False, total_epochs=None,
                 cwd=None):
        self.args = args
        self.out_dir = out_dir
        self.has_phenix = has_phenix
        self.total_epochs = total_epochs
        self.cwd = cwd
        self.started = False
        self.resumed = False
        self.shut_down = False
        self.log = ""
        self._pid = 4242
        FakeRunner.instances.append(self)
        FakeRunner.events.append(("constructed", self))

    @classmethod
    def attach(cls, out_dir, has_phenix=False):
        return cls.attach_result

    def start(self):
        self.started = True

    def resume(self):
        self.resumed = True

    def shutdown(self, timeout=5.0):
        self.shut_down = True
        FakeRunner.events.append(("shutdown", self))

    def _append_log(self, text):
        self.log += text

    def to_widget(self):
        import ipywidgets as widgets

        return widgets.HTML(value="<i>fake runner</i>")


@pytest.fixture
def fake_runner(monkeypatch):
    """argparse_gui imports AbismalRunner *inside* run_abismal, so this reaches it."""
    FakeRunner.instances = []
    FakeRunner.attach_result = None
    FakeRunner.events = []
    monkeypatch.setattr("abismal.gui.runner.AbismalRunner", FakeRunner)
    return FakeRunner


def filled(form, out_dir):
    H.set_control(form, "inputs", "a.mtz")
    H.set_control(form, "dmin", "1.8")
    H.set_control(form, "out_dir", str(out_dir))
    return form


def test_run_launches_a_runner_with_the_forms_arguments(tiny_form, fake_runner, tmp_path):
    filled(tiny_form, tmp_path)

    tiny_form.run_button.click()

    assert len(fake_runner.instances) == 1
    runner = fake_runner.instances[0]
    assert runner.started
    assert runner.out_dir == str(tmp_path)
    assert runner.total_epochs == 12
    # to_args emits option_strings[0], and dmin is declared as ("-d", "--dmin")
    assert "-d" in runner.args and "1.8" in runner.args


def test_a_relative_out_dir_is_resolved_before_launching(
    tiny_form, fake_runner, tmp_path, monkeypatch
):
    """The kernel and the child disagree about ".": the child is launched with
    cwd=default_directory() while the kernel sits wherever the .ipynb does. The
    runner opens console.log itself, so a relative out_dir would leave it
    watching a different directory than the one being written to."""
    monkeypatch.setattr(
        "abismal.gui.components.argparse_gui.default_directory",
        lambda: str(tmp_path),
    )
    H.set_control(tiny_form, "inputs", "a.mtz")
    H.set_control(tiny_form, "dmin", "1.8")
    H.set_control(tiny_form, "out_dir", "results")

    tiny_form.run_button.click()

    runner = fake_runner.instances[0]
    assert runner.out_dir == str(tmp_path / "results")
    assert runner.cwd == str(tmp_path)


def test_the_runners_widget_is_appended_to_the_form(tiny_form, fake_runner, tmp_path):
    filled(tiny_form, tmp_path)
    before = len(tiny_form.widget.children)

    tiny_form.run_button.click()

    assert len(tiny_form.widget.children) == before + 1


# `_on_run_click` writes inside `with self._run_output:`. An ipywidgets Output only
# captures when there is an IPython shell to redirect, so headlessly `.outputs` stays
# empty and the text goes to the real stdout instead. These assert on stdout, which
# checks the half that matters here -- that the failure is reported rather than
# swallowed, and that the click does not propagate an exception. Whether it lands in
# the widget rather than the terminal is a frontend behaviour, and belongs to the
# browser tier.

def test_invalid_arguments_are_reported_not_swallowed(
    tiny_form, fake_runner, tmp_path, capsys
):
    """argparse raises SystemExit. Uncaught on Colab, the click silently does nothing
    and the usage text goes nowhere at all."""
    H.set_control(tiny_form, "out_dir", str(tmp_path))   # no inputs, no dmin

    tiny_form.run_button.click()   # must not raise

    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "usage" in text or "required" in text
    assert not fake_runner.instances


def test_an_exception_is_reported_not_swallowed(
    tiny_form, fake_runner, tmp_path, monkeypatch, capsys
):
    def explode(*a, **k):
        raise RuntimeError("boom from the runner")

    monkeypatch.setattr(FakeRunner, "start", explode)
    filled(tiny_form, tmp_path)

    tiny_form.run_button.click()   # must not raise

    captured = capsys.readouterr()
    assert "boom from the runner" in captured.out + captured.err


# ---------------------------------------------------------------------------
# overwrite confirmation
# ---------------------------------------------------------------------------

def test_existing_output_asks_before_overwriting(tiny_form, fake_runner, tmp_path):
    (tmp_path / "history.csv").write_text("Epoch\n1\n")
    filled(tiny_form, tmp_path)
    before = len(tiny_form.widget.children)

    tiny_form.run_button.click()

    assert not fake_runner.instances, "must not launch before the user confirms"
    assert len(tiny_form.widget.children) == before + 1
    confirm = tiny_form.widget.children[-1]
    html = " ".join(getattr(w, "value", "") for w in _iter(confirm))
    assert "history.csv" in html


def test_cancel_restores_the_form(tiny_form, fake_runner, tmp_path):
    (tmp_path / "history.csv").write_text("Epoch\n1\n")
    filled(tiny_form, tmp_path)
    before = tuple(tiny_form.widget.children)
    tiny_form.run_button.click()

    confirm = tiny_form.widget.children[-1]
    buttons = [w for w in _iter(confirm) if type(w).__name__ == "Button"]
    next(b for b in buttons if "cancel" in b.description.lower()).click()

    assert tuple(tiny_form.widget.children) == before
    assert not fake_runner.instances
    assert (tmp_path / "history.csv").exists()


def test_confirming_deletes_the_old_output_and_launches(tiny_form, fake_runner, tmp_path):
    (tmp_path / "history.csv").write_text("Epoch\n1\n")
    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()

    confirm = tiny_form.widget.children[-1]
    buttons = [w for w in _iter(confirm) if type(w).__name__ == "Button"]
    next(b for b in buttons if "overwrite" in b.description.lower()).click()

    assert not (tmp_path / "history.csv").exists()
    assert len(fake_runner.instances) == 1
    assert fake_runner.instances[0].started


# ---------------------------------------------------------------------------
# replacing a runner
# ---------------------------------------------------------------------------

# Swapping the runner's widget out of the form does not stop the runner behind it: it
# keeps a poll timer, a phenix watcher thread and -- on Colab -- a browser interval
# syncing its widgets, all of them reading an out_dir the next run is about to rewrite.


def test_relaunching_stops_the_runner_it_replaces(tiny_form, fake_runner, tmp_path):
    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()
    first = fake_runner.instances[0]

    tiny_form.run_button.click()

    assert len(fake_runner.instances) == 2
    assert first.shut_down
    assert not fake_runner.instances[1].shut_down


def test_reconnecting_stops_the_runner_it_replaces(tiny_form, fake_runner, tmp_path):
    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()
    first = fake_runner.instances[0]
    fake_runner.attach_result = FakeRunner(args=None, out_dir=str(tmp_path))

    tiny_form.run_button.click()

    assert first.shut_down


def test_the_previous_runner_is_stopped_before_its_output_is_deleted(
    tiny_form, fake_runner, tmp_path, monkeypatch
):
    """The ordering is the fix. A poll that overlaps the delete reads a result
    directory mid-rmtree and hands the 3D viewer files that are gone by the time the
    browser asks for them, which leaves it loading forever.
    """
    import abismal.gui.cleanup as cleanup_module

    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()
    first = fake_runner.instances[0]

    real_cleanup = cleanup_module.cleanup_abismal_outputs

    def spy(out_dir):
        fake_runner.events.append(("cleanup", out_dir))
        return real_cleanup(out_dir)

    monkeypatch.setattr(cleanup_module, "cleanup_abismal_outputs", spy)

    (tmp_path / "history.csv").write_text("Epoch\n1\n")
    tiny_form.run_button.click()
    confirm = tiny_form.widget.children[-1]
    buttons = [w for w in _iter(confirm) if type(w).__name__ == "Button"]
    next(b for b in buttons if "overwrite" in b.description.lower()).click()

    assert first.shut_down
    order = [name for name, _ in fake_runner.events]
    assert order.index("shutdown") < order.index("cleanup")


def test_a_second_run_stops_the_first_runner(tiny_form, fake_runner, tmp_path):
    """The overwrite dialog is not the only way to get two runners. Running into a
    directory with nothing in it swaps the widget straight out, and the runner behind
    it keeps its poll timer, its watcher thread and its Colab interval either way.
    """
    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()
    first = fake_runner.instances[0]

    tiny_form.run_button.click()
    second = fake_runner.instances[1]

    assert first.shut_down
    assert not second.shut_down
    assert tiny_form._runner is second


def test_reattaching_stops_the_previous_runner(tiny_form, fake_runner, tmp_path):
    """Reconnecting to an orphaned job replaces the displayed runner too, and the one
    it replaces has to be put down like any other -- it is still polling out_dir.
    """
    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()
    first = fake_runner.instances[0]

    attached = FakeRunner(args=None, out_dir=str(tmp_path))
    FakeRunner.attach_result = attached
    tiny_form.run_button.click()

    assert first.shut_down
    assert attached.resumed
    assert tiny_form._runner is attached


def test_a_runner_is_never_asked_to_shut_itself_down(tiny_form, fake_runner, tmp_path):
    """_install_runner guards on identity. Re-installing the same runner -- which the
    attach path can do -- must not stop the one it is about to display.
    """
    filled(tiny_form, tmp_path)
    tiny_form.run_button.click()
    runner = fake_runner.instances[0]

    tiny_form._install_runner(runner)

    assert not runner.shut_down


def _iter(widget):
    yield widget
    for child in getattr(widget, "children", None) or ():
        yield from _iter(child)


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def test_a_running_job_is_reconnected_rather_than_relaunched(
    tiny_form, fake_runner, tmp_path
):
    existing = FakeRunner(args=None, out_dir=str(tmp_path))
    FakeRunner.instances = []           # the form should not construct another
    FakeRunner.attach_result = existing
    filled(tiny_form, tmp_path)

    tiny_form.run_button.click()

    assert not fake_runner.instances, "attached, so nothing new should be constructed"
    assert existing.resumed
    assert "Reconnected to running process PID 4242" in existing.log


# ---------------------------------------------------------------------------
# the real parser -- invariants only, never golden text
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_form():
    return H.build_form()


def test_the_real_form_builds(real_form):
    assert len(real_form._all_args) > 50
    assert len(real_form.children) > 5


def test_no_skipped_action_leaked_into_the_real_form(real_form):
    dests = {a.dest for a in real_form._all_args}
    assert not dests & set(ArgparseGUIBase.skipped_actions)


def test_file_selectors_are_wired_to_the_right_options(real_form):
    by_dest = {a.dest: type(w).__name__ for a, w in real_form._all_args.items()}
    # `inputs` keeps the accumulating two-panel browser; everything else that
    # names a file gets the compact picker, chosen by the argument's type.
    assert by_dest["inputs"] == "ReflectionFileSelector"
    for dest in ("eff_files", "torchref_pdb", "r_free_mtz", "reference_mtz",
                 "out_dir", "posterior_init_file", "scale_init_file"):
        assert by_dest[dest] == "PathSelector", dest


def test_every_path_argument_gets_a_picker(real_form):
    """The dispatch is on the type, so a new path argument needs nothing else.

    This is the regression guard for the complaint that started it: --r-free-mtz
    was a bare text box, leaving no way to discover which file to name.
    """
    from abismal.gui.components.file_selector import PathSelector, ServerFileSelectorWidget

    for action, widget in real_form._all_args.items():
        if action.type in ArgparseGUIBase.path_modes:
            assert isinstance(widget, (PathSelector, ServerFileSelectorWidget)), \
                f"{action.dest} names a path but rendered as {type(widget).__name__}"


def test_path_pickers_know_what_they_are_picking(real_form):
    modes = {
        a.dest: real_form._all_args[a].mode
        for a in real_form._all_args
        if type(real_form._all_args[a]).__name__ == "PathSelector"
    }
    # out_dir is the one save target: see tests/gui/test_out_dir.py
    assert modes["out_dir"] == "save"
    assert modes["r_free_mtz"] == "file"
    # --eff-files and --torchref-pdb are comma-separated lists on the CLI.
    assert modes["eff_files"] == "files"
    assert modes["torchref_pdb"] == "files"


def test_out_dir_lives_under_the_io_header(real_form):
    """It used to be hoisted next to the required arguments, which read as
    required. It is neither required nor special: it belongs with the rest of IO."""
    out_dir_widget = widget_for(real_form, "out_dir")
    assert out_dir_widget not in real_form.top_section.children
    assert out_dir_widget in real_form.children["IO"].children


def test_out_dir_is_prefilled_below_the_base_directory(real_form):
    """Its CLI default is ".", which in a notebook would put results next to the
    .ipynb rather than where the user is working. The rest of the row is covered
    in tests/gui/test_out_dir.py."""
    from abismal.gui.components.file_selector import default_directory

    assert widget_for(real_form, "out_dir").value.startswith(default_directory())


def test_required_options_are_promoted_out_of_the_tabs(real_form):
    required = {a.dest for a in real_form._all_args if a.required}
    in_top = {
        a.dest for a, w in real_form._all_args.items()
        if w in real_form.top_section.children
    }
    assert required <= in_top


def test_real_form_arguments_parse(real_form):
    """Whatever the form emits by default must be acceptable to the parser it came
    from, apart from the required options a user still has to fill in."""
    args = real_form.to_args()
    namespace = real_form.parser.parse_args(args + ["-d", "1.8", "in.mtz"])
    assert namespace.dmin == 1.8


def test_no_widget_class_colab_cannot_render(real_form):
    """Colab's widget manager cannot render these, which is why the group container is
    hand-rolled out of Buttons instead of using widgets.Tab."""
    forbidden = {"Tab", "Accordion", "Stack", "TagsInput", "FileUpload",
                 "AppLayout", "GridspecLayout", "TwoByTwoLayout"}
    seen = {type(w).__name__ for w in _iter(real_form.widget)}
    assert not seen & forbidden, sorted(seen & forbidden)
