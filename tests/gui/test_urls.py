"""_resolve_via_symlink and _files_url: turning a kernel-side path into a browser URL.

The 3D viewer runs in an iframe, so it fetches the pdb and mtz over HTTP rather than
reading them off disk. Getting the URL wrong means the viewer silently shows nothing,
which is exactly the failure that is expensive to diagnose by eye.

jupyter_server is not installed here, and deliberately so -- every use of it in the GUI
is ImportError-guarded. _files_url imports it *inside* the function, so a fake module
in sys.modules reaches it.
"""
import os
import sys
import types

import pytest

from abismal.gui.runner import _files_url, _resolve_via_symlink


@pytest.fixture
def fake_jupyter_server(monkeypatch):
    """Install a fake jupyter_server.serverapp whose server list is settable."""
    def install(servers):
        serverapp = types.ModuleType("jupyter_server.serverapp")
        serverapp.list_running_servers = lambda: list(servers)
        package = types.ModuleType("jupyter_server")
        package.serverapp = serverapp
        monkeypatch.setitem(sys.modules, "jupyter_server", package)
        monkeypatch.setitem(sys.modules, "jupyter_server.serverapp", serverapp)

    return install


# ---------------------------------------------------------------------------
# _resolve_via_symlink -- takes both paths as arguments, so it needs no faking
# ---------------------------------------------------------------------------

def test_symlink_pointing_straight_at_the_file(tmp_path):
    target = tmp_path / "real.mtz"
    target.write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(target, root / "link.mtz")

    assert _resolve_via_symlink(str(target), str(root)) == "link.mtz"


def test_symlink_pointing_at_an_ancestor_directory(tmp_path):
    """The shared-HPC case: $HOME is symlinked into the server root."""
    data = tmp_path / "scratch" / "run1"
    data.mkdir(parents=True)
    wanted = data / "refined.pdb"
    wanted.write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(tmp_path / "scratch", root / "scratch")

    assert _resolve_via_symlink(str(wanted), str(root)) == os.path.join(
        "scratch", "run1", "refined.pdb"
    )


def test_plain_directories_are_not_treated_as_links(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    wanted = root / "sub" / "f.mtz"
    wanted.write_text("x")

    assert _resolve_via_symlink(str(wanted), str(root)) is None


def test_no_match_returns_none(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "elsewhere.mtz"
    other.write_text("x")

    assert _resolve_via_symlink(str(other), str(root)) is None


def test_unreadable_root_is_none_not_an_exception(tmp_path):
    assert _resolve_via_symlink(str(tmp_path / "f"), str(tmp_path / "missing")) is None


# ---------------------------------------------------------------------------
# _files_url
# ---------------------------------------------------------------------------

def test_file_under_the_server_root(tmp_path, fake_jupyter_server):
    root = tmp_path / "root"
    (root / "run").mkdir(parents=True)
    wanted = root / "run" / "refined.pdb"
    wanted.write_text("x")
    fake_jupyter_server([{"root_dir": str(root), "url": "http://localhost:8888/"}])

    assert _files_url(str(wanted)) == "/files/run/refined.pdb"


def test_reverse_proxy_base_path_is_preserved(tmp_path, fake_jupyter_server):
    """Open OnDemand serves JupyterLab under a path prefix.

    The iframe inherits the proxy origin but not the server's base path, so a bare
    /files/... fetch would land outside the proxy and 404.
    """
    root = tmp_path / "root"
    root.mkdir()
    wanted = root / "refined.mtz"
    wanted.write_text("x")
    fake_jupyter_server([
        {"root_dir": str(root), "url": "https://ood.example/node/n01/8888/"}
    ])

    assert _files_url(str(wanted)) == "/node/n01/8888/files/refined.mtz"


def test_base_path_without_a_trailing_slash(tmp_path, fake_jupyter_server):
    root = tmp_path / "root"
    root.mkdir()
    wanted = root / "f.mtz"
    wanted.write_text("x")
    fake_jupyter_server([{"root_dir": str(root), "url": "https://h/node/n01/8888"}])

    assert _files_url(str(wanted)) == "/node/n01/8888/files/f.mtz"


def test_the_first_server_that_contains_the_file_wins(tmp_path, fake_jupyter_server):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    wanted = b / "f.mtz"
    wanted.write_text("x")
    fake_jupyter_server([
        {"root_dir": str(a), "url": "http://h:1/"},
        {"root_dir": str(b), "url": "http://h:2/"},
    ])

    assert _files_url(str(wanted)) == "/files/f.mtz"


def test_reaches_a_file_through_a_symlinked_root(tmp_path, fake_jupyter_server):
    data = tmp_path / "scratch" / "run1"
    data.mkdir(parents=True)
    wanted = data / "refined.pdb"
    wanted.write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(tmp_path / "scratch", root / "scratch")
    fake_jupyter_server([{"root_dir": str(root), "url": "http://localhost:8888/"}])

    assert _files_url(str(wanted)) == "/files/scratch/run1/refined.pdb"


def test_servers_without_a_root_dir_are_skipped(tmp_path, fake_jupyter_server):
    root = tmp_path / "root"
    root.mkdir()
    wanted = root / "f.mtz"
    wanted.write_text("x")
    fake_jupyter_server([{"url": "http://h/"}, {"root_dir": str(root), "url": "http://h/"}])

    assert _files_url(str(wanted)) == "/files/f.mtz"


def test_no_server_falls_back_to_a_cwd_relative_path(tmp_path, monkeypatch):
    """What happens on Colab, where jupyter_server does not exist.

    Colab does not serve /files/, so the viewer's fetches 404 there. Pinned as the
    current behaviour rather than endorsed -- see the Colab checklist.
    """
    wanted = tmp_path / "f.mtz"
    wanted.write_text("x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "jupyter_server", None)

    assert _files_url(str(wanted)) == "/files/f.mtz"
