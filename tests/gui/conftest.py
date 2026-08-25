"""Fixtures for the notebook GUI tests.

Everything here runs in a plain python process: no browser, no Jupyter kernel, no
Colab. That works because ipywidgets falls back to a no-op DummyComm when there is no
kernel, and because AbismalRunner._run_on_main_thread calls its closure synchronously
when get_ipython() returns None.
"""
import os
import threading
import warnings

# Before abismal is imported anywhere below: TF's cuFFT/cuDNN registration errors bury
# real failure output, and matplotlib must not go looking for a display.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

# The GUI's dependencies live in the optional [gui] extra, so a stock `pip install -e
# .[dev]` checkout cannot run any of this. Skip there -- but never skip silently in CI,
# where the workflow sets ABISMAL_REQUIRE_GUI_TESTS and a missing dependency should be
# a hard failure rather than a suite that quietly shrinks to nothing.
try:
    import ipywidgets  # noqa: F401
except ImportError:  # pragma: no cover - depends on how the env was installed
    if os.environ.get("ABISMAL_REQUIRE_GUI_TESTS"):
        raise
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    """Collapse the runner's timers.

    poll_interval is a class attribute, so this reaches every runner built during a
    test. It also shrinks _post_training_phenix_watcher, which waits for 12 unchanged
    polls -- 120 s at the default, 0.6 s here.
    """
    from abismal.gui.runner import AbismalRunner

    monkeypatch.setattr(AbismalRunner, "poll_interval", 0.05)


@pytest.fixture
def runner_factory(tmp_path):
    """Build AbismalRunners and guarantee they are put down afterwards.

    Never construct AbismalRunner directly in a test. Nothing in it cancels the
    self-rescheduling poll Timer, and with has_phenix=True _monitoring_active is never
    cleared, so a runner left alone keeps a timer chain and a watcher thread running
    into the next test. Replay children are started with start_new_session=True and so
    do not die with pytest either.
    """
    import gui_harness as H

    built = []

    def make(**kwargs):
        from abismal.gui.runner import AbismalRunner

        kwargs.setdefault("args", None)
        kwargs.setdefault("out_dir", str(tmp_path))
        kwargs.setdefault("has_phenix", False)
        runner = AbismalRunner(**kwargs)
        built.append(runner)
        return runner

    def adopt(runner):
        """Register a runner that something else built, e.g. start_replay."""
        built.append(runner)
        return runner

    make.adopt = adopt
    yield make

    for runner in built:
        try:
            H.quiesce(runner, timeout=5.0)
        except Exception as error:  # teardown must not mask the test's own failure
            warnings.warn(f"could not quiesce runner: {error!r}")


@pytest.fixture(autouse=True)
def usable_cwd(tmp_path):
    """Guarantee the working directory exists.

    The file selectors scan the cwd when they are constructed, and _files_url falls
    back to a cwd-relative path, so a deleted cwd makes them raise FileNotFoundError
    a long way from whatever deleted it. tests/command_line/test_cli.py used to leave
    exactly that behind; this keeps the GUI suite independent of test ordering.
    """
    import os

    try:
        os.getcwd()
    except OSError:
        os.chdir(tmp_path)
    yield


@pytest.fixture(autouse=True)
def no_thread_leaks():
    """Warn about threads a test leaves behind.

    A warning rather than a failure for now: the goal is to notice a new leak, not to
    fail the suite over an existing one. Promote to an assert once the suite is clean.
    """
    before = {t.ident for t in threading.enumerate()}
    yield
    leaked = [
        t for t in threading.enumerate()
        if t.ident not in before and t.is_alive() and not t.daemon
    ]
    if leaked:
        warnings.warn(f"test leaked non-daemon threads: {[t.name for t in leaked]}")


@pytest.fixture
def tiny_parser():
    """A small parser covering every branch of ArgparseGUIBase.action_to_widget.

    ArgparseGUIBase takes `parser=`, so the form can be built over this instead of
    abismal's 82-action singleton. It buys no import time -- argparse_gui imports the
    real parser at module scope regardless -- but it makes assertions readable and
    stable while the real CLI's options churn.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="tiny", add_help=False)
    parser.add_argument("inputs", nargs="+")                       # required positional
    parser.add_argument("-d", "--dmin", required=True)             # required option
    parser.add_argument("-o", "--out-dir", default="out")          # held aside by name
    parser.add_argument("--layers", default=5, type=int)           # plain store
    parser.add_argument("--activation", default="relu",
                        choices=("relu", "swish"))                 # store with choices
    parser.add_argument("--lower", default="Wilson", type=str.lower,
                        choices=("wilson", "normal"))              # default needs coercion
    parser.add_argument("--anomalous", action="store_true")        # store_true
    parser.add_argument("--no-cache", action="store_false")        # store_false
    # _launch_runner reads parsed.epochs to size the progress bar, so the run-path
    # tests need it even though it is not interesting to the widget dispatch.
    parser.add_argument("--epochs", default=12, type=int)
    # A second group, so there is more than one panel to switch between. The GUI keys
    # its panels off group.title, and the tab container is the hand-rolled Colab
    # workaround worth having a regression test for.
    extras = parser.add_argument_group("Extras")
    extras.add_argument("--seed", default=0, type=int)
    extras.add_argument("--verbose", action="store_true")
    return parser


@pytest.fixture
def tiny_form(tiny_parser):
    """A built form over :func:`tiny_parser`.

    Built on ArgparseGUIBase, not ArgparseGUI: the latter maps `inputs` to a
    ReflectionFileSelector, and the tiny parser's `inputs` is an ordinary positional.
    The selector mapping is asserted against the real parser instead.
    """
    import gui_harness as H
    from abismal.gui.components.argparse_gui import ArgparseGUIBase

    return H.build_form(parser=tiny_parser, cls=ArgparseGUIBase)


@pytest.fixture
def replay_dir():
    """Where the captured console.log / history.csv fixtures live."""
    import gui_harness as H

    return H.REPLAY_DIR
