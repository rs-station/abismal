"""The one dependency abismal cannot declare.

dxtbx and dials are not installable from PyPI -- what is published under those
names is a 2020 source-only dxtbx and a 0.0.1 dials placeholder -- so there is
no extra to write. The requirement is enforced where it is used instead.
"""
import pytest

from abismal.io import _dials_check


@pytest.fixture
def without_dials(monkeypatch):
    """Hide dxtbx and dials without disturbing any other lookup.

    Patched on `_dials_check`, not on importlib: the module does
    `from importlib.util import find_spec`, so it holds its own reference and
    patching the source has no effect. The same by-value binding already caught
    us with `_is_colab` in the GUI.
    """
    real = _dials_check.find_spec

    def fake(name, *args, **kwargs):
        if name in ("dxtbx", "dials"):
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(_dials_check, "find_spec", fake)


def test_missing_dials_names_the_conda_command(without_dials):
    """A bare ModuleNotFoundError does not say the fix is a different installer.

    That is how it surfaced on Colab: a traceback ending in `No module named
    'dxtbx'`, several frames below anything the user wrote, with no indication
    that pip cannot supply it.
    """
    with pytest.raises(SystemExit) as excinfo:
        _dials_check.require_dials(["a.expt", "a.refl"])

    message = str(excinfo.value)
    assert "conda" in message
    assert "dxtbx" in message and "dials" in message
    # It must also say who does *not* need this, or every stream user who hits
    # it will think they have a broken install.
    assert ".stream" in message and ".mtz" in message


def test_the_offending_files_are_listed(without_dials):
    with pytest.raises(SystemExit) as excinfo:
        _dials_check.require_dials(["one.expt", "one.refl"])
    assert "one.expt" in str(excinfo.value)


def test_a_long_file_list_is_truncated(without_dials):
    """A stills dataset can be thousands of files; the message must stay readable."""
    files = [f"chunk{n:04d}.expt" for n in range(500)]
    with pytest.raises(SystemExit) as excinfo:
        _dials_check.require_dials(files)
    message = str(excinfo.value)
    assert "and 497 more" in message
    assert message.count("chunk") == 3


def test_it_is_silent_when_dials_is_present():
    if not _dials_check.dials_is_available():
        pytest.skip("dials is not installed in this environment")
    assert _dials_check.require_dials(["a.expt"]) is None


def test_no_dials_extra_is_declared():
    """A `[dials]` extra would resolve to dead PyPI packages and look like it worked.

    If one is ever added, it has to be because real wheels appeared -- check
    PyPI before deleting this.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    from pathlib import Path

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not running from a checkout")

    data = tomllib.loads(pyproject.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "dials" not in extras
    for name, deps in extras.items():
        for dep in deps:
            assert not dep.startswith(("dxtbx", "dials")), (
                f"extra {name!r} lists {dep!r}, which PyPI cannot supply"
            )
