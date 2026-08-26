"""The whole runner path, driven by a replayed abismal.

These launch a real subprocess through AbismalRunner.start(). Only the executable is
substituted -- tests/gui/replay/abismal replays a captured console log and grows a
history.csv as it goes. Everything else is the shipped code: subprocess.Popen,
start_new_session, the stdout redirect into console.log, the pid file, /proc liveness,
the exit code, the tailer thread and the poll timer.

That is deliberate. is_running, _tail's exit condition, _schedule_poll's early return
and the pid-file lifecycle are all driven by _pid_is_abismal reading
/proc/<pid>/cmdline, so a Popen mock would force mocking that too and most of what
these tests exist to cover would be stub talking to stub. The stub's cmdline contains
`abismal`, so the real check passes for the right reason.
"""
import os
import time
from pathlib import Path

import pytest

import gui_harness as H

pytestmark = pytest.mark.skipif(
    not Path("/proc").is_dir(),
    reason="_pid_is_abismal reads /proc/<pid>/cmdline",
)


def wait_until(predicate, timeout=15.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def replay(tmp_path, runner_factory):
    """Start a replay and register it for teardown."""
    def start(**kwargs):
        kwargs.setdefault("results", H.make_results_template(tmp_path / "_template"))
        runner = H.start_replay(tmp_path / "run", **kwargs)
        return runner_factory.adopt(runner)

    return start


# ---------------------------------------------------------------------------
# the whole path
# ---------------------------------------------------------------------------

def test_a_complete_run(replay):
    """The flagship: launch, tail, progress, history, results, exit -- in a second or
    two, with no GPU and no merge job."""
    runner = replay(total_epochs=12)
    H.wait_for_replay(runner, timeout=30)

    log = H.log_text(runner)
    assert "oneDNN custom operations are on" in log      # first line of the fixture
    assert "Epoch 12/12" in log                          # last epoch
    assert log.index("Epoch 1/12") < log.index("Epoch 12/12")

    assert runner.progress_widget.max == 12
    assert runner.progress_widget.value == 12
    assert runner.progress_widget.bar_style == "success"
    assert runner.progress_label.value == "Finished"
    assert runner.stop_button.disabled is True

    pngs = H.extract_pngs(runner.history_widget)
    assert pngs and pngs[0][1][:4] == b"\x89PNG"

    pdb_file, _ = runner._find_latest_phenix_results()
    assert pdb_file and "epoch_12" in pdb_file


def test_the_log_is_escaped_on_the_way_in(replay):
    """The captured log contains tab-indented `[[{{node IteratorGetNext}}]]` blocks."""
    runner = replay()
    H.wait_for_replay(runner, timeout=30)

    assert "[[{{node" in H.log_text(runner)
    assert "&lt;" in runner.log_widget.value or "<pre" in runner.log_widget.value


def test_output_streams_rather_than_arriving_at_the_end(replay):
    """This is the automated form of "did it launch and are the logs moving?".

    Sampling from the test thread while the child is still writing shows the log
    growing and the progress bar advancing.
    """
    runner = replay(delay=0.03, total_epochs=12)

    samples = []
    for _ in range(12):
        time.sleep(0.25)
        samples.append((runner.progress_widget.value, len(runner.log_widget.value)))
        if not runner.is_running:
            break
    H.wait_for_replay(runner, timeout=30)

    log_lengths = [n for _, n in samples]
    progress = [p for p, _ in samples]
    assert log_lengths[-1] > log_lengths[0], "the log never grew"
    assert all(b >= a for a, b in zip(log_lengths, log_lengths[1:])), "log shrank"
    assert all(b >= a for a, b in zip(progress, progress[1:])), "progress went backwards"


def test_the_history_plot_is_redrawn_during_the_run(replay):
    """The stub appends a history row per epoch, so the plot has to change mid-run.

    Sampled until a second render appears rather than across a fixed window: one
    figure costs ~80 ms to draw and the replay takes a couple of seconds, so a
    window wide enough on an idle machine is a coin toss on a loaded one.
    """
    runner = replay(delay=0.05)

    seen = set()
    deadline = time.time() + 30
    while time.time() < deadline and len(seen) < 2:
        for _, png in H.extract_pngs(runner.history_widget):
            seen.add(len(png))
        if not runner.is_running and seen:
            break
        time.sleep(0.05)
    H.wait_for_replay(runner, timeout=30)
    for _, png in H.extract_pngs(runner.history_widget):
        seen.add(len(png))

    assert len(seen) > 1, "the plot never changed while the job was running"


def test_per_epoch_results_are_picked_up_as_they_appear(replay):
    runner = replay(delay=0.02)
    H.wait_for_replay(runner, timeout=30)

    assert runner._viewer_initialized
    assert runner._last_pdb is not None
    assert "Epoch 12" in runner.viewer_label.value


def test_the_first_epoch_renders_an_iframe_and_later_ones_reload_it(replay):
    """Re-embedding the viewer per epoch would reset the camera, so subsequent epochs
    postMessage into the existing iframe instead."""
    runner = replay(delay=0.02)
    H.wait_for_replay(runner, timeout=30)

    viewer_payloads = [
        item.get("data", {}) for item in (runner.viewer_widget.outputs or ())
    ]
    assert any("text/html" in payload for payload in viewer_payloads)

    scripts = [js for _, js in H.extract_scripts(runner._js_widget)]
    assert scripts, "no reload payload was emitted for the later epochs"
    assert "postMessage" in scripts[0]
    assert runner._viewer_id in scripts[0]


# ---------------------------------------------------------------------------
# how a run ends
# ---------------------------------------------------------------------------

def test_a_failing_run_is_reported_as_failed(replay):
    runner = replay(exit_code=1)
    H.wait_for_replay(runner, timeout=30)

    assert runner.progress_widget.bar_style == "danger"
    assert "Failed" in runner.progress_label.value
    assert "1" in runner.progress_label.value


def test_the_pid_file_is_written_then_cleaned_up(replay):
    runner = replay(delay=0.03)
    assert wait_until(lambda: os.path.exists(runner.pid_file)), "no pid file appeared"

    H.wait_for_replay(runner, timeout=30)

    assert not os.path.exists(runner.pid_file)


def test_stop_terminates_a_running_job(replay):
    runner = replay(hang=True)
    assert wait_until(lambda: runner.is_running), "the child never came up"

    runner.stop_button.click()

    assert wait_until(lambda: not runner.is_running, timeout=15)
    H.wait_for_replay(runner, timeout=15)
    assert runner.stop_button.disabled is True


def test_stop_escalates_to_sigkill(replay):
    """The stop path sends SIGTERM, waits 20 x 0.5 s, then SIGKILL. A child that
    ignores SIGTERM is the only way to reach the second half, and the 10 s wait is
    hard-coded, so this is the one test here that cannot be fast."""
    runner = replay(hang=True, ignore_sigterm=True)
    assert wait_until(lambda: runner.is_running), "the child never came up"

    runner.stop_button.click()

    assert wait_until(lambda: not runner.is_running, timeout=25)


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def test_attach_reconnects_to_a_job_this_process_did_not_start(replay, tmp_path):
    """What happens when the kernel is restarted under a running job."""
    from abismal.gui.runner import AbismalRunner

    original = replay(hang=True, delay=0.03)
    assert wait_until(lambda: os.path.exists(original.pid_file))
    original.shutdown()          # stop monitoring, leave the child running

    attached = AbismalRunner.attach(str(tmp_path / "run"))
    assert attached is not None
    assert attached._pid == original._pid

    attached.resume()
    assert wait_until(lambda: "Epoch" in H.log_text(attached), timeout=20)

    attached.shutdown()
    original.stop_button.click()
    wait_until(lambda: not original.is_running, timeout=15)


@pytest.mark.xfail(
    strict=True,
    reason="_tail opens console.log with no existence guard, so attaching to a job "
           "whose log is not there yet kills the tailer daemon thread with an "
           "unhandled FileNotFoundError -- silently, since nothing joins it.",
)
def test_attach_survives_a_missing_console_log(tmp_path, monkeypatch, runner_factory):
    """The thread dying is itself the bug, so this has to watch for the exception.

    Asserting the thread is no longer alive would pass either way -- it is not alive
    precisely because it crashed. threading.excepthook is what catches that; without
    it the traceback goes nowhere, since the thread is a daemon nobody joins.
    """
    import threading

    from abismal.gui.runner import AbismalRunner

    (tmp_path / "abismal.pid").write_text("12345")
    monkeypatch.setattr(AbismalRunner, "_pid_is_abismal", staticmethod(lambda pid: True))

    crashes = []
    monkeypatch.setattr(threading, "excepthook", lambda args: crashes.append(args))

    runner = runner_factory.adopt(AbismalRunner.attach(str(tmp_path)))
    runner.resume()
    wait_until(lambda: not runner._tailer_thread.is_alive(), timeout=5)
    runner.shutdown()

    assert not crashes, f"tailer thread died: {crashes[0].exc_value!r}"


# ---------------------------------------------------------------------------
# the anomalous peak plot, during a live run
# ---------------------------------------------------------------------------

def test_the_peak_plot_fills_in_while_the_run_goes(replay):
    """The template carries a peaks.csv and the stub scales its peakz per epoch,
    so this exercises the same path a real anomalous refinement takes: files
    appearing under out_dir one epoch at a time, picked up by the poll."""
    runner = replay(total_epochs=12)
    H.wait_for_replay(runner, timeout=30)

    peaks = runner._read_peaks()
    assert peaks is not None
    assert peaks["Epoch"].nunique() == 12
    assert set(peaks["Residue"]) >= {"CYS-30:A", "MET-105:A"}

    # And the peak heights actually climb, so the plot has a shape.
    first = peaks[peaks["Epoch"] == 1].set_index("Residue")["peakz"]
    last = peaks[peaks["Epoch"] == 12].set_index("Residue")["peakz"]
    assert (last > first).all()

    assert runner.peaks_widget.outputs
    assert runner.peaks_label.layout.display == ""


def test_no_peak_plot_without_refinement(replay):
    """A plain merge writes no per-epoch directories at all."""
    runner = replay(total_epochs=12, has_phenix=False, results=None)
    H.wait_for_replay(runner, timeout=30)

    assert runner._read_peaks() is None
    assert not runner.peaks_widget.outputs
    assert runner.peaks_label.layout.display == "none"


# ---------------------------------------------------------------------------
# re-running into the same directory
# ---------------------------------------------------------------------------

def test_overwriting_leaves_no_results_for_the_next_run_to_show(
    replay, tmp_path, runner_factory
):
    """The reported bug, end to end. cleanup_abismal_outputs matched `eff_*` but not
    `torchref_*`, while _find_latest_phenix_results globs both -- so after `Overwrite
    and Run` the runner the form built next found the previous job's results and
    rendered them as its own before the new job had written anything at all.
    """
    from abismal.gui.cleanup import cleanup_abismal_outputs, find_abismal_outputs

    runner = replay(total_epochs=12, prefix="torchref")
    H.wait_for_replay(runner, timeout=30)
    assert "epoch_12" in (runner._last_pdb or "")
    runner.shutdown()

    out_dir = str(tmp_path / "run")
    listed = find_abismal_outputs(out_dir)
    assert any("torchref_0_asu_0_epoch_12" in p for p in listed), \
        "the confirmation dialog never warned about these"
    cleanup_abismal_outputs(out_dir)

    fresh = runner_factory(out_dir=out_dir, has_phenix=True)
    assert fresh._find_latest_phenix_results() == (None, None)
    fresh._update_viewer()
    assert not fresh.viewer_widget.outputs


# ---------------------------------------------------------------------------
# the child's working directory
# ---------------------------------------------------------------------------

def test_the_child_is_launched_in_the_given_directory(tmp_path, runner_factory):
    """The form passes default_directory() so that a relative path typed into any
    field means the same thing to the child as it does to the picker that
    offered it -- not whatever directory the .ipynb happens to live in."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    runner = runner_factory.adopt(
        H.start_replay(tmp_path / "run", cwd=elsewhere, total_epochs=2)
    )
    H.wait_for_replay(runner, timeout=30)

    assert (tmp_path / "run" / "cwd.txt").read_text().strip() == str(elsewhere)


def test_no_cwd_means_inherit(tmp_path, runner_factory):
    """A bare AbismalRunner must behave as it always did."""
    runner = runner_factory.adopt(
        H.start_replay(tmp_path / "run", total_epochs=2)
    )
    H.wait_for_replay(runner, timeout=30)

    assert (tmp_path / "run" / "cwd.txt").read_text().strip() == os.getcwd()
