"""The compact path picker, and what relative paths resolve against.

Both are driven entirely by widget traits and the filesystem, so none of this
needs a kernel: `Button.click()` runs its handlers synchronously, and the browser
is shown and hidden by setting `layout.display`, which reads straight back.
"""
import os
import subprocess
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

needs_proc = pytest.mark.skipif(
    not os.path.isdir("/proc/self"), reason="reads the server's cwd from /proc"
)

# Above /proc/sys/kernel/pid_max on any sane system, so /proc/<it>/cwd cannot
# exist and cannot be a recycled pid either.
NO_SUCH_PID = 2 ** 30


@pytest.fixture
def server_process():
    """Start a process with a chosen cwd, to stand in for a jupyter server.

    What makes the launch directory knowable is the server process's own cwd, so
    a fake server has to be a real process sitting in a real directory.
    """
    started = []

    def start(directory):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            cwd=str(directory), stdin=subprocess.PIPE,
        )
        started.append(proc)
        return proc.pid

    yield start

    for proc in started:
        proc.kill()
        proc.wait()


def fake_servers(monkeypatch, *infos):
    fake = type(sys)("jupyter_server.serverapp")
    fake.list_running_servers = lambda: list(infos)
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", fake)
    monkeypatch.setitem(sys.modules, "jupyter_server", type(sys)("jupyter_server"))


@needs_proc
def test_default_directory_is_where_jupyter_was_launched(
    monkeypatch, tmp_path, server_process
):
    """`jupyter lab checkout/abismal_gui.ipynb` puts root_dir *and* the kernel's
    cwd in the checkout, which is what dropped results next to the .ipynb. The
    server process stays in the directory it was started from."""
    launched_from = tmp_path / "data"
    checkout = tmp_path / "opt" / "abismal"
    checkout.mkdir(parents=True)
    launched_from.mkdir()
    pid = server_process(launched_from)
    fake_servers(monkeypatch, {"pid": pid, "root_dir": str(checkout)})
    monkeypatch.setenv("JPY_PARENT_PID", str(pid))
    monkeypatch.chdir(checkout)

    assert default_directory() == str(launched_from.resolve())


@needs_proc
def test_the_server_that_spawned_this_kernel_wins(
    monkeypatch, tmp_path, server_process
):
    """More than one server can be running, and JPY_PARENT_PID says outright
    which one this kernel belongs to."""
    theirs = tmp_path / "theirs"
    ours = tmp_path / "ours"
    theirs.mkdir()
    ours.mkdir()
    our_pid = server_process(ours)
    fake_servers(
        monkeypatch,
        {"pid": server_process(theirs), "root_dir": str(theirs)},
        {"pid": our_pid, "root_dir": str(ours)},
    )
    monkeypatch.setenv("JPY_PARENT_PID", str(our_pid))

    assert default_directory() == str(ours.resolve())


@needs_proc
def test_without_a_parent_pid_the_deepest_containing_root_wins(
    monkeypatch, tmp_path, server_process
):
    """Nested roots both contain the kernel, so the deepest is the one that
    actually spawned it."""
    outer_root = tmp_path / "outer"
    inner_root = outer_root / "inner"
    inner_root.mkdir(parents=True)
    outer_launch = tmp_path / "outer-launch"
    inner_launch = tmp_path / "inner-launch"
    outer_launch.mkdir()
    inner_launch.mkdir()
    fake_servers(
        monkeypatch,
        {"pid": server_process(outer_launch), "root_dir": str(outer_root)},
        {"pid": server_process(inner_launch), "root_dir": str(inner_root)},
    )
    monkeypatch.delenv("JPY_PARENT_PID", raising=False)
    monkeypatch.chdir(inner_root)

    assert default_directory() == str(inner_launch.resolve())


def test_pwd_stands_in_where_proc_cannot_be_read(monkeypatch, tmp_path):
    """No /proc off Linux. PWD is the shell's directory at launch, inherited by
    the server and through it by this kernel, so it survives the same trip."""
    launched_from = tmp_path / "data"
    launched_from.mkdir()
    fake_servers(monkeypatch, {"pid": NO_SUCH_PID, "root_dir": str(tmp_path)})
    monkeypatch.setenv("JPY_PARENT_PID", str(NO_SUCH_PID))
    monkeypatch.setenv("PWD", str(launched_from))

    assert default_directory() == str(launched_from.resolve())


def test_the_server_root_is_the_last_resort(monkeypatch, tmp_path):
    """With nothing to identify the kernel's server, root_dir is the only thing
    left -- right whenever jupyter lab was started with no file argument."""
    root = tmp_path / "root"
    root.mkdir()
    fake_servers(monkeypatch, {"pid": NO_SUCH_PID, "root_dir": str(root)})
    monkeypatch.delenv("JPY_PARENT_PID", raising=False)

    assert default_directory() == str(root.resolve())


def test_pwd_is_ignored_outside_a_kernel(monkeypatch, tmp_path):
    """Nothing spawned us, so PWD is just our own cwd under another name -- and
    a stale one if anything has chdir'd since."""
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", None)
    monkeypatch.delenv("JPY_PARENT_PID", raising=False)
    monkeypatch.setenv("PWD", "/somewhere/stale")
    monkeypatch.chdir(tmp_path)

    assert Path(default_directory()).resolve() == tmp_path.resolve()


def test_falling_back_to_the_cwd_without_a_server(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", None)
    monkeypatch.delenv("JPY_PARENT_PID", raising=False)
    monkeypatch.chdir(tmp_path)

    assert Path(default_directory()).resolve() == tmp_path.resolve()


def test_a_bare_selector_opens_in_the_base_directory(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", None)
    monkeypatch.delenv("JPY_PARENT_PID", raising=False)
    monkeypatch.chdir(tmp_path)

    selector = PathSelector(description="x")

    assert selector._current_dir == tmp_path.resolve()


def test_the_reflection_selector_honours_its_initial_directory(tree):
    """Its __init__ used to swallow every argument via a bare super().__init__(),
    so it always opened on the kernel's cwd whatever it was handed."""
    from abismal.gui.components.file_selector import ReflectionFileSelector

    selector = ReflectionFileSelector(initial_directory=tree)

    assert selector._current_dir == tree.resolve()
