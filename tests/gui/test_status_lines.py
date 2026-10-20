"""The GUI's progress text and refinement list, from the run's [status] and
[refinement] lines and the refinement workers' own logs."""
import gui_harness as H  # noqa: F401  (sets up the widget environment the runner needs)

from abismal.gui.runner import _refinement_stage


def _worker_log(directory, text):
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("stdout.txt").write_text(text)


def test_the_stage_follows_the_worker_log(tmp_path):
    d = tmp_path / "torchref_0_asu_0_epoch_3"
    assert _refinement_stage(d) == "starting"
    _worker_log(d, "Loaded 2306 atoms\nBuilding restraints...\n")
    assert _refinement_stage(d) == "setting up"
    _worker_log(d, "fitting SIMU deviatoric sigma over [1.0, 0.5]\n    sigma=1      Rwork=0.12 Rfree=0.15\n")
    assert _refinement_stage(d) == "fitting the ADP restraint weight"
    _worker_log(d, "  rigid-body d_min=6.00 (lbfgs, iters=30): Rwork=0.27 Rfree=0.24\n")
    assert _refinement_stage(d) == "rigid-body refinement"
    _worker_log(d, "Cycle  2/5: Rwork=0.1395  Rfree=0.1464  [simu]\n")
    assert _refinement_stage(d) == "macrocycle 2/5 (R<sub>free</sub> 0.1464)"
    _worker_log(d, "Cycle  5/5: Rwork=0.13  Rfree=0.14\nfound 10 anomalous peaks above 5.0 sigma\n")
    assert _refinement_stage(d) == "finding anomalous peaks"


def test_status_lines_reach_the_progress_panel(runner_factory, tmp_path):
    runner = runner_factory(out_dir=str(tmp_path))
    runner._handle_line("[status] Compiling the model; the first steps take a minute or two\n")
    assert "Compiling the model" in runner.status_label.value
    runner._handle_line("[status] Training\n")
    assert "Training" in runner.status_label.value
    runner._handle_line("Epoch 1/30\n")          # not a status line
    assert "Training" in runner.status_label.value


def test_running_refinements_are_listed_with_their_stage(runner_factory, tmp_path):
    runner = runner_factory(out_dir=str(tmp_path), has_phenix=True)
    _worker_log(tmp_path / "torchref_0_asu_0_epoch_2", "Cycle  3/5: Rwork=0.2  Rfree=0.2154\n")
    runner._handle_line("[refinement] started torchref_0_asu_0_epoch_1\n")
    runner._handle_line("[refinement] finished torchref_0_asu_0_epoch_1\n")
    runner._handle_line("[refinement] started torchref_0_asu_0_epoch_2\n")
    runner._handle_line("[refinement] skipped torchref_0_asu_0_epoch_3 (1 still running)\n")
    shown = runner.refinement_status.value
    assert "epoch 2</b>: macrocycle 3/5" in shown
    assert "Finished: 1 (latest epoch 1)" in shown
    assert "Skipped while another was running: epoch 3" in shown
    assert "epoch 1</b>" not in shown          # finished ones are summarised, not listed


def test_a_failed_refinement_points_at_its_log(runner_factory, tmp_path):
    runner = runner_factory(out_dir=str(tmp_path), has_phenix=True)
    runner._handle_line("[refinement] started torchref_0_asu_0_epoch_4\n")
    runner._handle_line("[refinement] failed torchref_0_asu_0_epoch_4\n")
    shown = runner.refinement_status.value
    assert "Failed: epoch 4" in shown and "stderr.txt" in shown
    assert "No refinement running" in shown


def test_a_skip_does_not_hide_a_refinement_that_ran(runner_factory, tmp_path):
    """The final epoch can be skipped and then refined at the end of training."""
    runner = runner_factory(out_dir=str(tmp_path), has_phenix=True)
    runner._handle_line("[refinement] started torchref_0_asu_0_epoch_5\n")
    runner._handle_line("[refinement] skipped torchref_0_asu_0_epoch_5 (1 still running)\n")
    assert "epoch 5</b>" in runner.refinement_status.value
