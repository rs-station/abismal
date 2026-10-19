"""--torchref-pdb without torchref installed must fail before training.

The refinement worker is launched with Popen as a detached subprocess and its
stderr is redirected to a file nothing reads, so an ImportError in there stops
nothing. The run trains to completion, writes every mtz, and silently refines
nothing -- and the only evidence is the absence of directories the user has no
particular reason to expect. That is worth a preflight check.
"""
import importlib.util
from os import chdir, getcwd
from tempfile import TemporaryDirectory

import pytest

from abismal.command_line.abismal import main as abismal_main


base_flags = (
    "--dmin=5.0",
    "--epochs=1",
    "--batch-size=3",
    "--d-model=4",
    "--layers=2",
    "--shuffle-buffer-size=10",
    "--steps-per-epoch=2",
    "--num-cpus=1",
)


@pytest.fixture
def torchref_missing(monkeypatch):
    """Hide torchref from find_spec without disturbing any other lookup.

    Patching find_spec wholesale would make every later import in the run fail,
    which is a different bug than the one under test.
    """
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "torchref":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_missing_torchref_stops_the_run(conventional_mtz, torchref_missing, tmp_path):
    pdb = tmp_path / "model.pdb"
    pdb.write_text("END\n")

    original_cwd = getcwd()
    try:
        with TemporaryDirectory() as out:
            chdir(out)
            args = list(base_flags) + [
                "-o", "./",
                f"--torchref-pdb={pdb}",
                conventional_mtz,
            ]
            with pytest.raises(SystemExit) as excinfo:
                abismal_main(args)
    finally:
        chdir(original_cwd)

    message = str(excinfo.value)
    assert "torchref" in message
    assert "abismal[torchref]" in message, message
