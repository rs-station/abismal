"""The compact path picker, and what relative paths resolve against.

Both are driven entirely by widget traits and the filesystem, so none of this
needs a kernel: `Button.click()` runs its handlers synchronously, and the browser
is shown and hidden by setting `layout.display`, which reads straight back.
"""
import os
import sys
from pathlib import Path

import pytest

from abismal.gui.components.file_selector import (
    PathSelector,
    default_directory,
)


@pytest.fixture
def tree(tmp_path):
    """A small directory to browse: two subdirectories and a few files."""
    (tmp_path / "hewl").mkdir()
    (tmp_path / "thermolysin").mkdir()
    (tmp_path / ".hidden").mkdir()
    for name in ("free.mtz", "merged.mtz", "model.pdb", "notes.txt"):
        (tmp_path / name).write_text("")
    (tmp_path / "hewl" / "inner.mtz").write_text("")
    return tmp_path


# ---------------------------------------------------------------------------
# the compact row
# ---------------------------------------------------------------------------

def test_the_browser_is_closed_until_asked_for(tree):
    """Seven of these render at once. Any of them open by default would make the
    form unusable, which is why this is not the tall two-panel selector."""
    selector = PathSelector(description="r_free_mtz", initial_directory=tree)

    assert not selector.browser_open
    assert selector._browser.layout.display == "none"
    assert selector._browse_button.description == "Browse"


def test_browse_toggles_the_browser(tree):
    selector = PathSelector(description="r_free_mtz", initial_directory=tree)

    selector._browse_button.click()
    assert selector.browser_open
    assert selector._browse_button.description == "Close"

    selector._browse_button.click()
    assert not selector.browser_open
    assert selector._browse_button.description == "Browse"


def test_the_field_is_the_value_so_typing_still_works(tree):
    """Browsing is a way to fill the field in, not a replacement for it. Anyone
    who already knows the path should not have to click through to it."""
    selector = PathSelector(description="r_free_mtz", initial_directory=tree)

    selector._input.value = "/somewhere/else/free.mtz"

    assert selector.value == "/somewhere/else/free.mtz"


def test_selecting_a_file_fills_the_field_and_closes_the_browser(tree):
    selector = PathSelector(description="r_free_mtz", initial_directory=tree)
    selector._browse_button.click()

    selector._file_list.value = "free.mtz"
    selector._select_button.click()

    assert selector.value == str(tree / "free.mtz")
    assert not selector.browser_open


def test_selecting_nothing_leaves_the_browser_open(tree):
    """Clicking Select with no file highlighted is a slip, not a choice to clear
    the field."""
    selector = PathSelector(description="r_free_mtz", initial_directory=tree)
    selector._input.value = "keep/me.mtz"
    selector._browse_button.click()

    selector._select_button.click()

    assert selector.value == "keep/me.mtz"
    assert selector.browser_open


def test_cancel_closes_without_touching_the_field(tree):
    selector = PathSelector(description="r_free_mtz", initial_directory=tree)
    selector._input.value = "keep/me.mtz"
    selector._browse_button.click()
    selector._file_list.value = "free.mtz"

    selector._cancel_button.click()

    assert selector.value == "keep/me.mtz"
    assert not selector.browser_open


# ---------------------------------------------------------------------------
# the three modes
# ---------------------------------------------------------------------------

def test_files_mode_joins_with_commas(tree):
    """--eff-files and --torchref-pdb take a comma-separated list, so the widget
    has to produce exactly what the CLI parses."""
    selector = PathSelector(description="eff_files", mode="files",
                            initial_directory=tree)
    selector._browse_button.click()

    selector._file_list.value = ("free.mtz", "merged.mtz")
    selector._select_button.click()

    assert selector.value == f"{tree/'free.mtz'},{tree/'merged.mtz'}"


def test_files_mode_round_trips_through_the_parser(tree):
    from abismal.command_line.parser.custom_types import list_of_paths

    selector = PathSelector(description="eff_files", mode="files",
                            initial_directory=tree)
    selector._browse_button.click()
    selector._file_list.value = ("free.mtz", "merged.mtz")
    selector._select_button.click()

    assert list_of_paths(selector.value) == [tree / "free.mtz", tree / "merged.mtz"]


def test_directory_mode_takes_where_you_navigated_to(tree):
    """out_dir names a directory that need not exist yet, so there is nothing to
    pick out of a file list."""
    selector = PathSelector(description="out_dir", mode="directory",
                            initial_directory=tree)
    selector._browse_button.click()
    assert selector._file_list is None

    selector._dir_list.value = "hewl"
    selector._select_button.click()

    assert selector.value == str(tree / "hewl")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        PathSelector(description="x", mode="folder")


# ---------------------------------------------------------------------------
# navigation and filtering
# ---------------------------------------------------------------------------

def test_only_matching_suffixes_are_offered(tree):
    selector = PathSelector(description="r_free_mtz", file_types=(".mtz",),
                            initial_directory=tree)
    selector._browse_button.click()

    assert set(selector._file_list.options) == {"free.mtz", "merged.mtz"}


def test_no_filter_shows_everything(tree):
    selector = PathSelector(description="anything", initial_directory=tree)
    selector._browse_button.click()

    assert "notes.txt" in selector._file_list.options


def test_hidden_directories_are_not_listed(tree):
    selector = PathSelector(description="x", initial_directory=tree)
    selector._browse_button.click()

    assert ".hidden" not in selector._dir_list.options
    assert set(selector._dir_list.options) == {"hewl", "thermolysin"}


def test_clicking_a_directory_navigates_into_it(tree):
    selector = PathSelector(description="x", initial_directory=tree)
    selector._browse_button.click()

    selector._dir_list.value = "hewl"

    assert selector._current_dir == tree / "hewl"
    assert "inner.mtz" in selector._file_list.options


def test_up_goes_back(tree):
    selector = PathSelector(description="x", initial_directory=tree / "hewl")
    selector._browse_button.click()

    selector._up_button.click()

    assert selector._current_dir == tree


def test_reopening_starts_where_the_current_value_points(tree):
    """Correcting a typo in a deep path should not mean navigating to it again."""
    selector = PathSelector(description="x", initial_directory=tree)
    selector._input.value = str(tree / "hewl" / "inner.mtz")

    selector._browse_button.click()

    assert selector._current_dir == tree / "hewl"


def test_reopening_on_a_nonexistent_path_does_not_raise(tree):
    selector = PathSelector(description="x", initial_directory=tree)
    selector._input.value = "/no/such/place/at/all.mtz"

    selector._browse_button.click()

    assert selector.browser_open


def test_an_unreadable_directory_is_survivable(tree):
    """A shared filesystem has directories you cannot list. Browsing into one
    should do nothing, not kill the click handler."""
    locked = tree / "locked"
    locked.mkdir()
    os.chmod(locked, 0o000)
    try:
        selector = PathSelector(description="x", initial_directory=tree)
        selector._browse_button.click()
        selector._dir_list.value = "locked"
        assert selector.browser_open
    finally:
        os.chmod(locked, 0o755)


# ---------------------------------------------------------------------------
# where relative paths resolve
# ---------------------------------------------------------------------------

def test_default_directory_prefers_the_jupyter_root(monkeypatch, tmp_path):
    """Not the kernel's cwd, which for the shipped notebook is the abismal
    checkout -- that is what put results next to the .ipynb."""
    root = tmp_path / "data"
    root.mkdir()
    fake = type(sys)("jupyter_server.serverapp")
    fake.list_running_servers = lambda: [{"root_dir": str(root)}]
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", fake)
    monkeypatch.setitem(sys.modules, "jupyter_server", type(sys)("jupyter_server"))

    assert default_directory() == str(root.resolve())


def test_the_deepest_containing_root_wins(monkeypatch, tmp_path):
    """Two servers can be running, and one root can nest inside another. The one
    this kernel is actually inside is the useful answer."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    fake = type(sys)("jupyter_server.serverapp")
    fake.list_running_servers = lambda: [
        {"root_dir": str(outer)}, {"root_dir": str(inner)},
    ]
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", fake)
    monkeypatch.setitem(sys.modules, "jupyter_server", type(sys)("jupyter_server"))
    monkeypatch.chdir(inner)

    assert default_directory() == str(inner.resolve())


def test_falling_back_to_the_cwd_without_a_server(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", None)
    monkeypatch.chdir(tmp_path)

    assert Path(default_directory()).resolve() == tmp_path.resolve()


def test_a_bare_selector_opens_in_the_base_directory(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", None)
    monkeypatch.chdir(tmp_path)

    selector = PathSelector(description="x")

    assert selector._current_dir == tmp_path.resolve()


def test_the_reflection_selector_honours_its_initial_directory(tree):
    """Its __init__ used to swallow every argument via a bare super().__init__(),
    so it always opened on the kernel's cwd whatever it was handed."""
    from abismal.gui.components.file_selector import ReflectionFileSelector

    selector = ReflectionFileSelector(initial_directory=tree)

    assert selector._current_dir == tree.resolve()
