"""`abismal.gui` -- copy the packaged notebook here and start JupyterLab on it.

Nothing here execs jupyter for real: os.execv is patched so the command that
would have replaced this process is inspectable instead.
"""
import os
import stat
import sys
from pathlib import Path

import pytest

from abismal.command_line.gui import (
    DEFAULT_NAME,
    find_jupyter,
    main,
    packaged_notebook,
    place_notebook,
)


@pytest.fixture
def fake_exec(monkeypatch):
    """Capture the command main() would have exec'd."""
    calls = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, argv)))
    return calls


@pytest.fixture
def fake_jupyter(monkeypatch, tmp_path):
    """An executable named `jupyter` beside a fake sys.executable."""
    bindir = tmp_path / "env" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("")
    jupyter = bindir / "jupyter"
    jupyter.write_text("#!/bin/sh\n")
    jupyter.chmod(jupyter.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    return jupyter


# ---------------------------------------------------------------------------
# the notebook ships
# ---------------------------------------------------------------------------

def test_the_notebook_is_inside_the_package():
    """At the repo root it was not packaged at all, so `pip install abismal[gui]`
    installed JupyterLab and the widgets but nothing to open in them."""
    notebook = packaged_notebook()

    assert notebook.is_file()
    assert notebook.name == DEFAULT_NAME
    assert notebook.parent.name == "gui"


def test_the_packaged_notebook_is_a_valid_notebook():
    import nbformat

    nb = nbformat.read(str(packaged_notebook()), as_version=4)

    nbformat.validate(nb)
    assert len(nb.cells) >= 2


def test_the_packaged_notebook_carries_no_outputs():
    """It is a template. Shipping someone's execution output in it would put a
    stale TensorFlow banner and a stranger's file paths in front of every user.
    """
    import nbformat

    nb = nbformat.read(str(packaged_notebook()), as_version=4)

    assert not any(cell.get("outputs") for cell in nb.cells)


# ---------------------------------------------------------------------------
# placing the copy
# ---------------------------------------------------------------------------

def test_it_copies_the_notebook_into_the_working_directory(tmp_path):
    destination, copied = place_notebook(tmp_path / DEFAULT_NAME)

    assert copied
    assert destination.read_bytes() == packaged_notebook().read_bytes()


def test_an_existing_notebook_is_left_alone(tmp_path):
    """By the second run it holds the user's own edits and outputs. Replacing
    those because they typed the same command again would be indefensible."""
    destination = tmp_path / DEFAULT_NAME
    destination.write_text("my own work")

    returned, copied = place_notebook(destination)

    assert not copied
    assert returned.read_text() == "my own work"


def test_fresh_replaces_it(tmp_path):
    destination = tmp_path / DEFAULT_NAME
    destination.write_text("my own work")

    _, copied = place_notebook(destination, fresh=True)

    assert copied
    assert destination.read_bytes() == packaged_notebook().read_bytes()


def test_a_missing_package_notebook_is_a_clear_error(tmp_path, monkeypatch):
    import abismal.command_line.gui as gui

    monkeypatch.setattr(gui, "packaged_notebook", lambda: tmp_path / "gone.ipynb")

    with pytest.raises(SystemExit, match="packaged notebook is missing"):
        gui.place_notebook(tmp_path / DEFAULT_NAME)


# ---------------------------------------------------------------------------
# finding jupyter
# ---------------------------------------------------------------------------

def test_the_interpreters_own_jupyter_wins(fake_jupyter, monkeypatch):
    """Console scripts run by absolute path, so the environment abismal lives in
    need not be on PATH at all -- and a different one's jupyter would start
    kernels that cannot import what the notebook needs."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/jupyter")

    assert find_jupyter() == str(fake_jupyter)


def test_it_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/jupyter")

    assert find_jupyter() == "/usr/bin/jupyter"


def test_a_non_executable_sibling_is_not_used(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "jupyter").write_text("not executable")
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    monkeypatch.setattr("shutil.which", lambda name: None)

    assert find_jupyter() is None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_it_launches_jupyter_on_the_copy(tmp_path, monkeypatch, fake_exec,
                                         fake_jupyter):
    monkeypatch.chdir(tmp_path)

    main([])

    (path, argv), = fake_exec
    assert path == str(fake_jupyter)
    assert argv[1:] == ["lab", DEFAULT_NAME]
    assert (tmp_path / DEFAULT_NAME).is_file()


def test_unknown_options_are_passed_through_to_jupyter(tmp_path, monkeypatch,
                                                       fake_exec, fake_jupyter):
    """So --port, --no-browser and the rest keep working without this having to
    know about any of them."""
    monkeypatch.chdir(tmp_path)

    main(["--no-browser", "--port", "9999"])

    (_, argv), = fake_exec
    assert argv[-3:] == ["--no-browser", "--port", "9999"]


def test_the_notebook_can_be_named(tmp_path, monkeypatch, fake_exec, fake_jupyter):
    monkeypatch.chdir(tmp_path)

    main(["--notebook", "hewl_run.ipynb"])

    (_, argv), = fake_exec
    assert argv[2] == "hewl_run.ipynb"
    assert (tmp_path / "hewl_run.ipynb").is_file()


def test_print_path_does_not_launch_anything(capsys, fake_exec):
    assert main(["--print-path"]) == 0

    assert not fake_exec
    assert capsys.readouterr().out.strip() == str(packaged_notebook())


def test_no_jupyter_says_how_to_get_one(tmp_path, monkeypatch, fake_exec):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(SystemExit, match=r'abismal\[gui\]'):
        main([])

    assert not fake_exec
