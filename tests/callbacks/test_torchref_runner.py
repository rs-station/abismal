"""Tests for TorchRefRunner, the callback that spawns the torchref worker.

None of these run torchref. They stub `Popen` and assert on the argv the
callback builds and on how it manages the resulting processes, which is where
the failures that actually bit us live: an option accepted by the worker but
never forwarded is invisible until someone reads a log and notices the default
was used.
"""
import sys
import types

import pytest

tfk = pytest.importorskip("tf_keras")

from abismal.callbacks.torchref import TorchRefRunner


class FakeProcess:
    """Stand-in for Popen: finishes when told to, records that it was waited on."""

    def __init__(self, returncode=0, running=True):
        self._returncode = returncode
        self._running = running
        self.waited = False

    def finish(self):
        self._running = False

    def poll(self):
        return None if self._running else self._returncode

    def wait(self):
        self.waited = True
        self._running = False
        return self._returncode

    @property
    def returncode(self):
        return None if self._running else self._returncode


@pytest.fixture
def spawned(monkeypatch):
    """Capture argv from every spawn; hand back the fake processes."""
    calls = []

    def fake_popen(command, **kwargs):
        process = FakeProcess()
        calls.append((command, process))
        return process

    monkeypatch.setattr("abismal.callbacks.torchref.Popen", fake_popen)
    return calls


def _runner(tmp_path, **kwargs):
    pdb = tmp_path / "model.pdb"
    pdb.write_text("END\n")
    kwargs.setdefault("pdb_file", str(pdb))
    return TorchRefRunner(str(tmp_path), **kwargs)


def _argv_value(argv, flag):
    return argv[argv.index(flag) + 1]


# --------------------------------------------------------------------------
# argv construction -- "accepted but never forwarded" is the bug class here
# --------------------------------------------------------------------------

def test_every_option_reaches_the_worker(tmp_path, spawned):
    rfree = tmp_path / "rfree.mtz"
    rfree.write_text("")
    runner = _runner(
        tmp_path, r_free_mtz=str(rfree), r_free_value=1, wavelength=1.54,
        adp_mode="anisotropic", adp_aniso_sigma="0.2", macro_cycles=7,
        z_score_cutoff=4.5, rigid_body_iter=100,
    )
    runner.run_torchref(0)
    argv, _ = spawned[0]

    assert _argv_value(argv, "--r-free-value") == "1"
    assert _argv_value(argv, "--wavelength") == "1.54"
    assert _argv_value(argv, "--adp-mode") == "anisotropic"
    assert _argv_value(argv, "--adp-aniso-sigma") == "0.2"
    assert _argv_value(argv, "--macro-cycles") == "7"
    assert _argv_value(argv, "--z-score-cutoff") == "4.5"
    assert _argv_value(argv, "--rigid-body-iter") == "100"
    assert _argv_value(argv, "--r-free-mtz").endswith("rfree.mtz")


def test_rigid_body_flag_only_when_disabled(tmp_path, spawned):
    _runner(tmp_path, rigid_body=True).run_torchref(0)
    assert "--no-rigid-body" not in spawned[0][0]

    _runner(tmp_path, rigid_body=False).run_torchref(0)
    assert "--no-rigid-body" in spawned[1][0]


def test_zero_wavelength_is_forwarded(tmp_path, spawned):
    """0 means "disable anomalous refinement", and 0 is falsy.

    A truthiness test here would silently drop the option and quietly re-enable
    anomalous refinement, so pin it.
    """
    _runner(tmp_path, wavelength=0.0).run_torchref(0)
    assert _argv_value(spawned[0][0], "--wavelength") == "0.0"


def test_r_free_value_needs_its_mtz(tmp_path, spawned):
    """`--r-free-value` alone is meaningless and must not be emitted."""
    _runner(tmp_path, r_free_value=1).run_torchref(0)
    argv = spawned[0][0]
    assert "--r-free-value" not in argv
    assert "--r-free-mtz" not in argv


# --------------------------------------------------------------------------
# process lifecycle
# --------------------------------------------------------------------------

def test_finished_workers_are_reaped(tmp_path, spawned):
    """Finished workers must be cleared, not merely capped by the skip guard.

    Asserting `len(processes) <= 1` would pass with reaping entirely disabled,
    because the overlap guard caps the list at 1 by itself. What distinguishes
    reaping is that the list reaches ZERO and that a spawn therefore happens on
    every epoch.
    """
    runner = _runner(tmp_path, epoch_stride=1)
    for epoch in range(4):
        runner.on_epoch_end(epoch)
        for _, process in spawned:
            process.finish()
        runner._reap()
        assert runner.processes == [], f"not reaped after epoch {epoch}"
    assert len(spawned) == 4, "a finished worker blocked the next epoch"


def test_overlapping_runs_are_skipped(tmp_path, spawned):
    """A refinement outliving its epoch must not stack up N deep."""
    runner = _runner(tmp_path, epoch_stride=1)
    runner.on_epoch_end(0)
    assert len(spawned) == 1
    with pytest.warns(RuntimeWarning, match="still going"):
        runner.on_epoch_end(1)
    assert len(spawned) == 1, "second run started while the first was live"


def test_worker_failure_is_surfaced(tmp_path, spawned):
    """A nonzero exit must not pass silently -- stderr.txt is inside a result
    directory nobody reads, and training otherwise reports success."""
    runner = _runner(tmp_path, epoch_stride=1)
    runner.on_epoch_end(0)
    process = spawned[0][1]
    process._returncode = 1
    process.finish()
    with pytest.warns(RuntimeWarning, match="exited with 1"):
        runner.on_epoch_end(1)


def test_train_end_waits_for_stragglers(tmp_path, spawned):
    runner = _runner(tmp_path, epoch_stride=1)
    runner.on_epoch_end(0)
    runner.on_train_end()
    assert spawned[0][1].waited
    assert runner.processes == []


def test_final_epoch_is_refined_even_when_skipped(tmp_path, spawned):
    """The last epoch must be refined even if the guard skipped it.

    Deliberately leaves the first worker RUNNING, so the guard skips every
    later epoch -- the case that matters. A version of this test that finishes
    and reaps each worker constructs the situation where nothing overlaps and
    would pass with the whole mechanism absent.

    This is the headline result: on measured timings a worker takes 1.33x an
    epoch on hewl, so the final epoch really can be skipped, and a multi-hour
    run would end with no refinement of the model it is judged on.
    """
    runner = _runner(tmp_path, epoch_stride=1)
    runner.on_epoch_end(0)                      # spawns, stays running
    assert len(spawned) == 1
    for epoch in (1, 2):
        with pytest.warns(RuntimeWarning, match="still going"):
            runner.on_epoch_end(epoch)
    assert len(spawned) == 1, "guard should have suppressed the middle epochs"

    runner.on_train_end()
    refined = [argv[argv.index("--mtz") + 1] for argv, _ in spawned]
    assert any("epoch_3" in m for m in refined), (
        f"final epoch never refined; only {refined}"
    )


def test_no_skip_means_no_catch_up(tmp_path, spawned):
    """A run whose final epoch already refined must not refine it twice."""
    runner = _runner(tmp_path, epoch_stride=1)
    for epoch in range(3):
        runner.on_epoch_end(epoch)
        for _, process in spawned:
            process.finish()
        runner._reap()
    before = len(spawned)
    runner.on_train_end()
    assert len(spawned) == before, "on_train_end re-ran an epoch that was not skipped"


def test_no_pdb_means_no_spawn(tmp_path, spawned):
    runner = TorchRefRunner(str(tmp_path), pdb_file=None)
    runner.on_epoch_end(0)
    assert spawned == []
