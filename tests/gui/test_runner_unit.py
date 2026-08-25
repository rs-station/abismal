"""AbismalRunner's pieces, driven directly.

Constructing a runner touches no disk -- it only builds widgets -- and
_run_on_main_thread calls its closure synchronously outside a kernel, so every method
here can be called straight from the test and the result read back off a widget.
"""
import os
import shutil

import pytest

import gui_harness as H


# ---------------------------------------------------------------------------
# log rendering
# ---------------------------------------------------------------------------

def test_log_is_html_escaped(runner_factory):
    """TF writes lines like `[[{{node IteratorGetNext}}]]` and paths with & in them.

    Unescaped, a stray < swallows the rest of the log into a bogus tag.
    """
    runner = runner_factory()
    runner._append_log("<script>alert(1)</script> & [[{{node IteratorGetNext}}]]\n")

    value = runner.log_widget.value
    assert "<script>" not in value
    assert "&lt;script&gt;" in value
    assert "&amp;" in value


def test_log_accumulates_in_order(runner_factory):
    runner = runner_factory()
    for line in ("first\n", "second\n", "third\n"):
        runner._append_log(line)

    text = H.log_text(runner)
    assert text.index("first") < text.index("second") < text.index("third")


def test_log_is_wrapped_for_display(runner_factory):
    runner = runner_factory()
    runner._append_log("hello\n")
    assert runner.log_widget.value.lstrip().startswith("<pre")


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------

def test_progress_reports_completed_epochs_not_the_current_one(runner_factory):
    """value = cur - 1 encodes "epoch `cur` is running, `cur-1` are done", so the bar
    only reaches max via the finish path. Pinned so a "fix" has to be deliberate."""
    runner = runner_factory(total_epochs=10)
    runner._update_progress(4, 10)

    assert runner.progress_widget.value == 3
    assert runner.progress_widget.max == 10
    assert runner.progress_label.value == "Epoch 4 / 10"


def test_progress_adopts_the_total_from_the_log(runner_factory):
    """attach() cannot know the epoch count, so the bar starts at max=1 and the first
    `Epoch n/m` line has to correct it."""
    runner = runner_factory(total_epochs=None)
    assert runner.progress_widget.max == 1

    runner._update_progress(1, 30)
    assert runner.progress_widget.max == 30


# ---------------------------------------------------------------------------
# history plot
# ---------------------------------------------------------------------------

def test_history_renders_a_png(runner_factory, tmp_path, replay_dir):
    shutil.copy(replay_dir / "history.csv", tmp_path / "history.csv")
    runner = runner_factory(out_dir=str(tmp_path))

    runner._update_history()

    pngs = H.extract_pngs(runner.history_widget)
    assert len(pngs) == 1
    assert pngs[0][1][:4] == b"\x89PNG"
    assert len(pngs[0][1]) > 5000


def test_history_replaces_rather_than_appends(runner_factory, tmp_path, replay_dir):
    """The poll timer calls this repeatedly; appending would grow without bound."""
    shutil.copy(replay_dir / "history.csv", tmp_path / "history.csv")
    runner = runner_factory(out_dir=str(tmp_path))

    runner._update_history()
    runner._update_history()

    assert len(runner.history_widget.outputs) == 1


@pytest.mark.parametrize("contents", [None, "", "Epoch,loss\n"])
def test_history_absent_or_empty_produces_nothing(runner_factory, tmp_path, contents):
    """The plot is polled while the run is still starting up, so the file is routinely
    missing or headerless. It must not raise on a background thread."""
    if contents is not None:
        (tmp_path / "history.csv").write_text(contents)
    runner = runner_factory(out_dir=str(tmp_path))

    runner._update_history()

    assert not runner.history_widget.outputs


def test_history_survives_a_malformed_file(runner_factory, tmp_path):
    (tmp_path / "history.csv").write_text("not,a\nvalid\x00csv\n")
    runner = runner_factory(out_dir=str(tmp_path))

    runner._update_history()  # must not raise


# ---------------------------------------------------------------------------
# per-epoch result discovery
# ---------------------------------------------------------------------------

def _epoch_dir(root, prefix, epoch, pdb=True, mtz=True, extra=None):
    directory = root / f"{prefix}_0_asu_0_epoch_{epoch}"
    directory.mkdir(parents=True, exist_ok=True)
    if pdb:
        (directory / "refined.pdb").write_text("x")
    if mtz:
        (directory / "refined.mtz").write_text("x")
    if extra:
        (directory / extra).write_text("x")
    return directory


@pytest.mark.parametrize("prefix", ["eff", "torchref"])
def test_finds_results_from_either_refinement_backend(runner_factory, tmp_path, prefix):
    _epoch_dir(tmp_path, prefix, 1)
    runner = runner_factory(out_dir=str(tmp_path))

    pdb_file, mtz_file = runner._find_latest_phenix_results()

    assert pdb_file and pdb_file.endswith("refined.pdb")
    assert mtz_file and mtz_file.endswith("refined.mtz")


def test_epochs_are_ordered_numerically_not_lexicographically(runner_factory, tmp_path):
    """'10' sorts before '2' as a string, so a lexicographic sort would stall the
    viewer on epoch 9 for the rest of the run."""
    for epoch in (2, 9, 10):
        _epoch_dir(tmp_path, "torchref", epoch)
    runner = runner_factory(out_dir=str(tmp_path))

    pdb_file, _ = runner._find_latest_phenix_results()

    assert "epoch_10" in pdb_file


def test_an_incomplete_latest_epoch_falls_back(runner_factory, tmp_path):
    """The newest directory is routinely half-written while refinement is running."""
    _epoch_dir(tmp_path, "torchref", 1)
    _epoch_dir(tmp_path, "torchref", 2, pdb=False)
    runner = runner_factory(out_dir=str(tmp_path))

    pdb_file, _ = runner._find_latest_phenix_results()

    assert "epoch_1" in pdb_file


def test_the_input_mtz_is_not_mistaken_for_a_result(runner_factory, tmp_path):
    directory = _epoch_dir(tmp_path, "torchref", 1, mtz=False)
    (directory / "input_data.mtz").write_text("x")
    (directory / "refined.mtz").write_text("x")
    runner = runner_factory(out_dir=str(tmp_path))

    _, mtz_file = runner._find_latest_phenix_results()

    assert mtz_file.endswith("refined.mtz")


def test_no_results_is_a_pair_of_nones(runner_factory, tmp_path):
    runner = runner_factory(out_dir=str(tmp_path))
    assert runner._find_latest_phenix_results() == (None, None)


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def test_attach_needs_a_pid_file(tmp_path):
    from abismal.gui.runner import AbismalRunner

    assert AbismalRunner.attach(str(tmp_path)) is None


def test_attach_ignores_an_unreadable_pid_file(tmp_path):
    from abismal.gui.runner import AbismalRunner

    (tmp_path / "abismal.pid").write_text("not a number")
    assert AbismalRunner.attach(str(tmp_path)) is None


def test_attach_ignores_a_pid_that_is_not_abismal(tmp_path, monkeypatch):
    from abismal.gui.runner import AbismalRunner

    (tmp_path / "abismal.pid").write_text("12345")
    monkeypatch.setattr(AbismalRunner, "_pid_is_abismal", staticmethod(lambda pid: False))

    assert AbismalRunner.attach(str(tmp_path)) is None


def test_attach_reconnects_to_a_live_job(tmp_path, monkeypatch, runner_factory):
    from abismal.gui.runner import AbismalRunner

    (tmp_path / "abismal.pid").write_text("12345")
    monkeypatch.setattr(AbismalRunner, "_pid_is_abismal", staticmethod(lambda pid: True))

    runner = AbismalRunner.attach(str(tmp_path))
    runner_factory.adopt(runner)

    assert runner is not None
    assert runner._pid == 12345
    assert runner.args is None      # nothing to relaunch from
    assert runner._process is None  # it is not our child


def test_pid_is_abismal_rejects_a_dead_process():
    from abismal.gui.runner import AbismalRunner

    assert AbismalRunner._pid_is_abismal(9_999_999) is False


def test_pid_is_abismal_accepts_this_process_when_it_matches():
    """/proc/<pid>/cmdline substring match. This interpreter's cmdline contains the
    repo path, so it matches -- which is itself worth knowing, since it means the check
    is loose enough to accept an unrelated process run from an abismal directory."""
    from abismal.gui.runner import AbismalRunner

    with open(f"/proc/{os.getpid()}/cmdline", "rb") as handle:
        expected = b"abismal" in handle.read()
    assert AbismalRunner._pid_is_abismal(os.getpid()) == expected


# ---------------------------------------------------------------------------
# widget assembly
# ---------------------------------------------------------------------------

def test_the_viewer_section_appears_only_with_refinement(runner_factory):
    plain = runner_factory(has_phenix=False).to_widget()
    with_viewer = runner_factory(has_phenix=True).to_widget()

    assert len(with_viewer.children) > len(plain.children)
    labels = " ".join(
        w.value for w in _iter(with_viewer) if isinstance(getattr(w, "value", None), str)
    )
    assert "Refinement Results" in labels


def test_a_plain_runner_allocates_no_viewer(runner_factory):
    runner = runner_factory(has_phenix=False)
    assert runner.viewer_widget is None
    assert runner._viewer_id is None


def test_construction_writes_nothing_to_disk(tmp_path, runner_factory):
    runner_factory(out_dir=str(tmp_path / "nonexistent"))
    assert not (tmp_path / "nonexistent").exists()


def _iter(widget):
    yield widget
    for child in getattr(widget, "children", None) or ():
        yield from _iter(child)


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

def test_shutdown_stops_monitoring(runner_factory):
    runner = runner_factory()
    runner._monitoring_active = True

    runner.shutdown()

    assert runner._monitoring_active is False


def test_shutdown_cancels_the_poll_timer(runner_factory):
    """_schedule_poll re-arms on every tick and only stops re-arming once is_running
    goes false, so an already-armed timer otherwise always survives."""
    import threading

    runner = runner_factory()
    fired = threading.Event()
    runner._poll_timer = threading.Timer(30.0, fired.set)
    runner._poll_timer.daemon = True
    runner._poll_timer.start()

    runner.shutdown()

    assert runner._poll_timer is None
    assert not fired.is_set()


def test_shutdown_is_idempotent(runner_factory):
    runner = runner_factory()
    runner.shutdown()
    runner.shutdown()  # must not raise


def test_shutdown_on_a_runner_that_never_started(runner_factory):
    runner = runner_factory()
    assert runner._poll_timer is None
    runner.shutdown()


def test_shutdown_leaves_the_subprocess_alone(runner_factory, monkeypatch):
    """An attached job is meant to outlive the kernel; stop() is what kills it."""
    import abismal.gui.runner as runner_module

    killed = []
    monkeypatch.setattr(runner_module.os, "kill", lambda pid, sig: killed.append(pid))
    runner = runner_factory()
    runner._pid = 12345

    runner.shutdown()

    assert not killed
    assert runner._pid == 12345
