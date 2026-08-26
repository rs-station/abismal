"""find_abismal_outputs / cleanup_abismal_outputs.

Pure filesystem logic, no widgets. This is what the overwrite-confirmation dialog
lists before it offers to delete a previous run, so an output it fails to report is
one the user is never warned about and that survives into the next run.
"""
import os

import pytest

from abismal.gui.cleanup import (
    _DIR_PATTERNS,
    _FILE_PATTERNS,
    cleanup_abismal_outputs,
    find_abismal_outputs,
)


@pytest.fixture
def populated(tmp_path):
    """One of everything abismal writes, plus decoys that must survive."""
    for name in ("abismal.log", "console.log", "abismal.pid", "datamanager.yml",
                 "history.csv", "epoch_0.keras", "epoch_10.keras", "asu_0_epoch_1.mtz"):
        (tmp_path / name).write_text("x")
    for name in ("eff_0_asu_0_epoch_1", "diffmaps_0"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "inner.txt").write_text("x")

    # Decoys: a user's own files, and a directory whose name merely starts the same way.
    (tmp_path / "user_notes.txt").write_text("keep me")
    (tmp_path / "epoch_notes.md").write_text("keep me")
    (tmp_path / "my_data").mkdir()
    (tmp_path / "my_data" / "precious.mtz").write_text("keep me")
    return tmp_path


def test_finds_every_known_output(populated):
    found = {os.path.basename(p) for p in find_abismal_outputs(populated)}
    assert "abismal.log" in found
    assert "history.csv" in found
    assert "epoch_0.keras" in found and "epoch_10.keras" in found
    assert "asu_0_epoch_1.mtz" in found
    assert "eff_0_asu_0_epoch_1" in found
    assert "diffmaps_0" in found


def test_leaves_unrelated_files_alone(populated):
    found = {os.path.basename(p) for p in find_abismal_outputs(populated)}
    assert "user_notes.txt" not in found
    assert "epoch_notes.md" not in found
    assert "my_data" not in found


def test_returns_sorted_absolute_paths(populated):
    found = find_abismal_outputs(populated)
    assert found == sorted(found)
    assert all(os.path.isabs(p) for p in found)


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert find_abismal_outputs(tmp_path / "nope") == []


def test_cleanup_removes_exactly_what_find_reported(populated):
    doomed = find_abismal_outputs(populated)
    cleanup_abismal_outputs(populated)

    assert not [p for p in doomed if os.path.exists(p)]
    # and nothing else went with them
    assert (populated / "user_notes.txt").exists()
    assert (populated / "epoch_notes.md").exists()
    assert (populated / "my_data" / "precious.mtz").exists()


def test_cleanup_of_a_clean_directory_is_a_noop(tmp_path):
    (tmp_path / "user_notes.txt").write_text("keep me")
    cleanup_abismal_outputs(tmp_path)
    assert (tmp_path / "user_notes.txt").exists()


def test_finds_torchref_result_directories(tmp_path):
    """_find_latest_phenix_results globs eff_* and torchref_* alike, so a torchref
    directory cleanup leaves behind is one the next run's viewer shows as its own.
    """
    (tmp_path / "torchref_0_asu_0_epoch_1").mkdir()
    (tmp_path / "torchref_0_asu_0_epoch_1" / "refined.pdb").write_text("x")

    found = {os.path.basename(p) for p in find_abismal_outputs(tmp_path)}
    assert "torchref_0_asu_0_epoch_1" in found


def test_a_users_own_torchref_directory_is_not_swept_up(tmp_path):
    """--torchref-pdb points at the user's own models, so a directory named after them
    beside out_dir is an ordinary thing to have. Hence the torchref pattern is the full
    `_asu_*_epoch_*` shape rather than the bare prefix `eff_*` gets.
    """
    (tmp_path / "torchref_models").mkdir()
    (tmp_path / "torchref_models" / "start.pdb").write_text("keep me")

    cleanup_abismal_outputs(tmp_path)

    assert (tmp_path / "torchref_models" / "start.pdb").exists()


def test_patterns_are_relative_not_globbed_paths():
    """A pattern with a separator in it would escape out_dir."""
    for pattern in _FILE_PATTERNS + _DIR_PATTERNS:
        assert os.sep not in pattern, pattern
