"""Telling a stale install from a stale notebook.

Both look like "the fix I just deployed isn't here", and they want opposite
remedies -- delete the runtime, or reload the page. Both have actually happened
while debugging Colab, which is why this exists.
"""
import json

import pytest

from abismal.gui import _build


A = "a" * 40
B = "b" * 40
C = "c" * 40


@pytest.fixture
def running(monkeypatch):
    """Pin what the installed build reports."""
    def set_to(commit):
        monkeypatch.setattr(_build, "installed_commit", lambda: commit)
    return set_to


@pytest.fixture
def head(monkeypatch):
    """Pin what the branch points at, including 'cannot ask'.

    A tip is a head plus its parents, because the deploy pushes the notebook as
    a commit on top of the code it pins.
    """
    def set_to(commit, parents=()):
        tip = None if commit is None else {"sha": commit, "parents": list(parents)}
        monkeypatch.setattr(_build, "branch_tip", lambda *a, **k: tip)
    return set_to


def test_a_stale_install_says_delete_the_runtime(running, head):
    """The notebook asked for one commit and something older is loaded."""
    running(A)
    head(B)
    message = _build.describe_build(expected=B, branch="colab-debug")
    assert "STALE" in message
    assert "delete runtime" in message
    assert A[:8] in message and B[:8] in message


def test_a_stale_notebook_says_reload(running, head):
    """The install matches the notebook, but the branch has moved past both."""
    running(A)
    head(C)
    message = _build.describe_build(expected=A, branch="colab-debug")
    assert "STALE" not in message, "the install is fine; the page is not"
    assert "Reload" in message
    assert C[:8] in message


def test_current_build_says_so(running, head):
    running(A)
    head(A)
    message = _build.describe_build(expected=A, branch="colab-debug")
    assert "current" in message


def test_the_notebook_commit_on_top_of_the_code_is_still_current(running, head):
    """The deploy pushes the notebook as a commit above the code it pins.

    So a healthy deploy always has installed == parent(head). Comparing against
    the head alone called every good deploy stale, which is worse than no check.
    """
    running(A)
    head(B, parents=[A])
    message = _build.describe_build(expected=A, branch="colab-debug")
    assert "current" in message, message
    assert "Reload" not in message


def test_an_unreachable_api_is_unknown_not_stale(running, head):
    """Rate limits are shared on Colab, so a failed lookup must not accuse."""
    running(A)
    head(None)
    message = _build.describe_build(expected=A, branch="colab-debug")
    assert "unknown" in message
    for alarming in ("STALE", "Reload", "delete runtime"):
        assert alarming not in message


def test_a_non_git_install_is_reported_honestly(monkeypatch):
    monkeypatch.setattr(_build, "installed_commit", lambda: None)
    assert "cannot be determined" in _build.describe_build(expected=A)


def test_installed_commit_reads_pip_s_record(monkeypatch):
    """pip writes the commit into direct_url.json for any VCS install."""
    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({
                "url": "https://github.com/rs-station/abismal",
                "vcs_info": {"vcs": "git", "commit_id": A},
            })

    import importlib.metadata as md
    monkeypatch.setattr(md, "distribution", lambda name: FakeDistribution())
    assert _build.installed_commit() == A


def test_a_wheel_without_vcs_info_is_not_a_commit(monkeypatch):
    """A local-path install records direct_url.json with no vcs_info at all."""
    class FakeDistribution:
        def read_text(self, name):
            return json.dumps({"dir_info": {}, "url": "file:///tmp/abismal-src"})

    import importlib.metadata as md
    monkeypatch.setattr(md, "distribution", lambda name: FakeDistribution())
    assert _build.installed_commit() is None
