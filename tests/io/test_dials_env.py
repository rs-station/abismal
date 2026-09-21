"""Installing the DIALS toolbox beside an interpreter that only has pip.

The constraints this encodes were measured, not guessed: prepending the prefix
shadows the host's ml_dtypes and breaks TensorFlow through jax, and leaving the
conda solve unconstrained picks a numpy that makes torchref uninstallable.
"""
import sys
from pathlib import Path

import pytest

from abismal.io import _dials_env as env


def test_the_solve_is_capped_below_numpy_2_4():
    """Unconstrained, conda picks numpy 2.5 and torchref stops being installable.

    torchref declares numpy<2.4.0; dxtbx unconstrained wants >=2.5.1. Capping
    the solve lands on dxtbx 3.26 with numpy 2.3.x, which TensorFlow accepts
    (>=1.26, no ceiling) and refinement still works alongside.
    """
    assert env.NUMPY_CEILING == "numpy<2.4"


def test_both_import_directories_are_used(tmp_path):
    """site-packages alone is not enough.

    The compiled boost extensions live in lib-dynload, and without it `import
    dials` fails on boost_python_meta_ext well after dxtbx imported cleanly.
    """
    site_packages, dynload = env._site_packages(tmp_path)
    assert site_packages.name == "site-packages"
    assert dynload.name == "lib-dynload"
    assert f"python{sys.version_info[0]}.{sys.version_info[1]}" in str(site_packages)


def test_activate_appends_and_never_prepends(tmp_path, monkeypatch):
    """Order is the whole ballgame.

    The prefix carries its own numpy, ml_dtypes, h5py and scipy. Ahead of the
    host's they shadow them, and conda's older ml_dtypes made jax raise on
    import -- taking TensorFlow with it. Appended, the prefix supplies only what
    is genuinely missing.
    """
    site_packages, dynload = env._site_packages(tmp_path)
    site_packages.mkdir(parents=True)
    dynload.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", ["/first", "/second"])
    monkeypatch.delenv("LIBTBX_BUILD", raising=False)

    assert env.activate(tmp_path) is True
    assert sys.path[:2] == ["/first", "/second"], "the prefix jumped the queue"
    assert str(site_packages) in sys.path
    assert str(dynload) in sys.path


def test_activate_sets_what_libtbx_and_the_linker_need(tmp_path, monkeypatch):
    """libtbx resolves a build path against sys.prefix, which is the host.

    Without LIBTBX_BUILD it goes looking for /usr/share/cctbx and raises.
    """
    site_packages, dynload = env._site_packages(tmp_path)
    site_packages.mkdir(parents=True)
    dynload.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("LIBTBX_BUILD", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env.activate(tmp_path)
    assert env.os.environ["LIBTBX_BUILD"] == str(tmp_path / "share" / "cctbx")
    assert str(tmp_path / "lib") in env.os.environ["LD_LIBRARY_PATH"]


def test_activate_is_idempotent(tmp_path, monkeypatch):
    """Called from dials_is_available, which anything may call repeatedly."""
    site_packages, dynload = env._site_packages(tmp_path)
    site_packages.mkdir(parents=True)
    dynload.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))

    env.activate(tmp_path)
    before = len(sys.path)
    env.activate(tmp_path)
    assert len(sys.path) == before, "activation duplicated its own entries"


def test_activate_is_a_no_op_with_nothing_configured(monkeypatch):
    """Everyone merging .mtz or .stream files takes this path and must pay nothing."""
    monkeypatch.delenv(env.PREFIX_ENV_VAR, raising=False)
    monkeypatch.setattr(env, "DEFAULT_PREFIX", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = list(sys.path)

    assert env.activate() is False
    assert sys.path == before


def test_activate_ignores_a_prefix_that_is_not_there(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert env.activate(tmp_path / "nope") is False


def test_the_env_var_overrides_the_default(tmp_path, monkeypatch):
    """So a cluster module or a hand-built prefix can be pointed at."""
    monkeypatch.setenv(env.PREFIX_ENV_VAR, str(tmp_path))
    assert env.configured_prefix() == tmp_path


def test_install_reuses_a_prefix_that_already_works(tmp_path, monkeypatch):
    """Several GB. It must never be fetched twice in one runtime."""
    site_packages, dynload = env._site_packages(tmp_path)
    site_packages.mkdir(parents=True)
    dynload.mkdir(parents=True)
    monkeypatch.setattr(env, "_micromamba",
                        lambda *a, **k: pytest.fail("micromamba was fetched"))
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert env.install(prefix=tmp_path, quiet=True) == tmp_path


def test_the_tarball_is_named_explicitly(tmp_path, monkeypatch):
    """Regression: tar read stdin and got nothing.

    This was translated from a shell `curl ... | tar -xj`, which reads stdin by
    design. Downloading to a file first means the archive has to be named, and
    without -f the extraction silently had no input.
    """
    calls = []
    monkeypatch.setattr(env.subprocess, "run",
                        lambda cmd, **k: calls.append(cmd) or type("R", (), {"returncode": 0})())

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url, dest: Path(dest).write_bytes(b""))
    monkeypatch.setattr(Path, "chmod", lambda self, mode: None)

    env._micromamba(tmp_path)
    assert calls, "tar was never run"
    assert "-f" in calls[0], f"no -f in {calls[0]}"
    assert str(tmp_path / "micromamba.tar.bz2") in calls[0]


def test_children_inherit_the_library_path(tmp_path, monkeypatch):
    """The subprocess is the path that has to work.

    The dynamic linker reads LD_LIBRARY_PATH when a process starts, so setting
    it inside a running interpreter does nothing for that interpreter -- the
    conda extensions want a newer CXXABI than an old host libstdc++ exports,
    and once TensorFlow has loaded the system one it is too late to preload
    another. A child starts fresh and reads the variable properly, and the
    merge runs in a child.
    """
    site_packages, dynload = env._site_packages(tmp_path)
    site_packages.mkdir(parents=True)
    dynload.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/pre-existing")

    env.activate(tmp_path)

    entries = env.os.environ["LD_LIBRARY_PATH"].split(":")
    assert str(tmp_path / "lib") in entries
    assert "/pre-existing" in entries, "an existing search path was discarded"


def test_a_prefix_is_not_built_off_linux(monkeypatch, capsys):
    """The mechanism is not portable, and claiming it would be worse than refusing.

    The workload runs in a subprocess that inherits LD_LIBRARY_PATH; macOS uses
    DYLD_LIBRARY_PATH and SIP strips it on exec, so the path that actually
    carries the merge would not survive.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(env, "_micromamba",
                        lambda *a, **k: pytest.fail("micromamba was fetched"))

    assert env.install(prefix="/tmp/nope", quiet=True) is None
    assert "conda install" in capsys.readouterr().out


def test_an_older_host_numpy_is_called_out(tmp_path, monkeypatch, capsys):
    """numpy's ABI is forward- but not backward-compatible.

    An extension built against 2.3 crashes on 2.0, and it crashes mid-parse
    rather than failing to import -- worth a sentence before it happens.
    """
    site_packages, dynload = env._site_packages(tmp_path)
    site_packages.mkdir(parents=True)
    dynload.mkdir(parents=True)
    (site_packages / "numpy-2.3.5.dist-info").mkdir()
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(env, "_preload_runtime", lambda prefix: None)

    import numpy
    monkeypatch.setattr(numpy, "__version__", "2.0.2")

    env.activate(tmp_path)
    out = capsys.readouterr().out
    assert "2.3.5" in out and "2.0.2" in out
