"""The setup a notebook used to carry, now that it is ordinary code.

It lived in a cell for a while: install logic, downloads and version checks that
could only be tested by matching strings against the cell's own source. Guards
written that way have passed while the code was wrong, and failed by matching
their own comments. These are tests of behaviour instead.
"""
import sys
import types

import pytest

from abismal.gui import _launch as launch_module


@pytest.fixture
def not_colab(monkeypatch):
    monkeypatch.setattr(launch_module, "on_colab", lambda: False)


@pytest.fixture
def no_downloads(monkeypatch):
    """Fail loudly if anything reaches the network."""
    def forbidden(*args, **kwargs):
        raise AssertionError("the network was touched")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


# ---------------------------------------------------------------------------
# example data is opt in
# ---------------------------------------------------------------------------

def test_launch_downloads_nothing_by_default(monkeypatch, no_downloads):
    """Someone who brought their own data should not be made to fetch 31 MB."""
    called = []
    monkeypatch.setattr(launch_module, "fetch_example_data",
                        lambda **k: called.append("example"))
    monkeypatch.setattr(launch_module, "fetch_reference_data",
                        lambda **k: called.append("reference"))
    monkeypatch.setattr(launch_module, "install_torchref",
                        lambda: called.append("torchref"))

    launch_module.launch(quiet=True)
    assert called == []


def test_example_data_brings_the_refinement_files_with_it(monkeypatch):
    """The reference data exists to drive refinement, so torchref comes too."""
    called = []
    monkeypatch.setattr(launch_module, "install_torchref",
                        lambda: called.append("torchref"))
    monkeypatch.setattr(launch_module, "fetch_example_data",
                        lambda **k: called.append("example") or "/content/x.mtz")
    monkeypatch.setattr(launch_module, "fetch_reference_data",
                        lambda **k: called.append("reference"))

    launch_module.launch(example_data=True, quiet=True)
    assert called == ["torchref", "example", "reference"]


def test_reference_data_is_skipped_when_the_reflections_fail(monkeypatch):
    """No use mirroring reference files for a dataset that never arrived."""
    called = []
    monkeypatch.setattr(launch_module, "install_torchref", lambda: None)
    monkeypatch.setattr(launch_module, "fetch_example_data", lambda **k: None)
    monkeypatch.setattr(launch_module, "fetch_reference_data",
                        lambda **k: called.append("reference"))

    launch_module.launch(example_data=True, quiet=True)
    assert called == []


def test_fetching_is_a_no_op_off_colab(not_colab, no_downloads):
    """/content is Colab. Nothing should land in a local working directory."""
    assert launch_module.fetch_example_data() is None
    assert launch_module.fetch_reference_data() is None


# ---------------------------------------------------------------------------
# torchref
# ---------------------------------------------------------------------------

def test_torchref_is_not_installed_without_torch(monkeypatch):
    """Otherwise this is an 800 MB surprise in someone's local environment."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: None if n in ("torch", "torchref")
                        else real(n, *a, **k))
    monkeypatch.setattr(launch_module.subprocess, "run",
                        lambda *a, **k: pytest.fail("pip was run"))

    assert launch_module.install_torchref() is False


def test_torchref_already_present_is_left_alone(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: object() if n in ("torch", "torchref")
                        else real(n, *a, **k))
    monkeypatch.setattr(launch_module.subprocess, "run",
                        lambda *a, **k: pytest.fail("pip was run"))

    assert launch_module.install_torchref() is True


# ---------------------------------------------------------------------------
# stale modules
# ---------------------------------------------------------------------------

def test_a_module_loaded_at_another_version_stops_the_launch(monkeypatch):
    """An install cannot replace a module the interpreter already imported.

    The old object stays in sys.modules and the new files sit unused, so the
    session runs code that is not installed. ipywidgets half a major out renders
    nothing at all, which is the hardest failure to read.
    """
    fake = types.ModuleType("ipywidgets")
    fake.__version__ = "7.7.1"
    monkeypatch.setitem(sys.modules, "ipywidgets", fake)

    import importlib.metadata as md
    monkeypatch.setattr(md, "version",
                        lambda name: "8.1.9" if name == "ipywidgets" else "1.0")

    assert launch_module.stale_modules(("ipywidgets",)) == [
        ("ipywidgets", "7.7.1", "8.1.9")
    ]
    with pytest.raises(RuntimeError, match="Restart the runtime"):
        launch_module.launch(quiet=True)


def test_matching_versions_are_not_stale(monkeypatch):
    fake = types.ModuleType("ipywidgets")
    fake.__version__ = "8.1.9"
    monkeypatch.setitem(sys.modules, "ipywidgets", fake)

    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "8.1.9")

    assert launch_module.stale_modules(("ipywidgets",)) == []


def test_a_module_that_was_never_imported_is_not_stale(monkeypatch):
    monkeypatch.delitem(sys.modules, "ipywidgets", raising=False)
    assert launch_module.stale_modules(("ipywidgets",)) == []


# ---------------------------------------------------------------------------
# what it returns
# ---------------------------------------------------------------------------

def test_launch_returns_the_form_for_the_notebook_to_display(monkeypatch, no_downloads):
    widget = launch_module.launch(quiet=True)
    assert type(widget).__name__ == "VBox"
    assert widget.children, "the form came back empty"


# ---------------------------------------------------------------------------
# output height
# ---------------------------------------------------------------------------

def test_launch_asks_colab_not_to_scroll_the_output(monkeypatch, no_downloads):
    """The panel is taller than Colab's output box, so it gets a scrollbar.

    You then land below the progress bar and the training history, which looks
    exactly like they failed to render -- and did, for a couple of rounds.
    """
    called = []
    fake_output = types.SimpleNamespace(
        no_vertical_scroll=lambda: called.append(True),
        enable_custom_widget_manager=lambda: None,
    )
    colab = types.ModuleType("google.colab")
    colab.output = fake_output
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.output", fake_output)

    launch_module.launch(quiet=True)
    assert called, "no_vertical_scroll was never called"


def test_no_output_scroll_is_a_no_op_off_colab(not_colab):
    assert launch_module.no_output_scroll() is False


def test_a_failure_to_unscroll_does_not_break_the_launch(monkeypatch, no_downloads):
    """Cosmetic. It must never be the thing that stops the GUI appearing."""
    def boom():
        raise RuntimeError("no frontend")

    fake_output = types.SimpleNamespace(no_vertical_scroll=boom,
                                        enable_custom_widget_manager=lambda: None)
    colab = types.ModuleType("google.colab")
    colab.output = fake_output
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.output", fake_output)

    assert launch_module.no_output_scroll() is False
    assert type(launch_module.launch(quiet=True)).__name__ == "VBox"
