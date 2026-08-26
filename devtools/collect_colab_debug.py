#!/usr/bin/env python
"""Print everything worth knowing when the GUI misbehaves on Colab.

Paste the output into the issue. Colab is the one environment that cannot be driven
from here, so this is the substitute for looking at it: it reports the facts that
distinguish the handful of things that actually go wrong there.

Run it in a cell of the notebook that is misbehaving, after the run has been started:

    !python -m pip download --no-deps -q abismal 2>/dev/null   # not needed; ignore
    from devtools.collect_colab_debug import report; report()

or, if devtools is not on the path (the usual case on Colab):

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/rs-station/abismal/gui/devtools/collect_colab_debug.py'
    ).read())
    report()
"""
from __future__ import annotations

import sys


def _version(name):
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception as error:
        return f"<{type(error).__name__}>"


def report(runner=None):
    """Print a diagnostic block. Pass the AbismalRunner if you have a handle on it."""
    lines = ["=== abismal GUI on Colab ==="]

    try:
        import google.colab  # noqa: F401

        on_colab = True
    except ImportError:
        on_colab = False
    lines.append(f"google.colab importable : {on_colab}")

    try:
        from abismal.gui.components.file_selector import _is_colab
        from abismal.gui import runner as runner_module

        lines.append(f"file_selector._is_colab(): {_is_colab()}")
        # runner imports _is_colab by value, so this is the one that decides whether
        # the polling loop was ever set up.
        lines.append(f"runner._is_colab()       : {runner_module._is_colab()}")
    except Exception as error:
        lines.append(f"could not probe _is_colab: {error!r}")

    lines.append(f"python                   : {sys.version.split()[0]}")
    for package in ("abismal", "ipywidgets", "solara", "ipython", "tensorflow",
                    "torch", "torchref", "reciprocalspaceship"):
        lines.append(f"{package:25s}: {_version(package)}")

    try:
        from IPython import get_ipython

        shell = get_ipython()
        kernel = getattr(shell, "kernel", None)
        lines.append(f"get_ipython()            : {type(shell).__name__ if shell else None}")
        lines.append(f"kernel has io_loop       : {hasattr(kernel, 'io_loop')}")
    except Exception as error:
        lines.append(f"could not probe IPython  : {error!r}")

    import threading

    alive = [t.name for t in threading.enumerate()]
    lines.append(f"threads ({len(alive)})            : {alive}")

    if runner is not None:
        lines.append("--- runner ---")
        lines.append(f"pid                      : {runner._pid}")
        lines.append(f"is_running               : {runner.is_running}")
        lines.append(f"_monitoring_active       : {runner._monitoring_active}")
        lines.append(f"poll timer armed         : {runner._poll_timer is not None}")
        tailer = runner._tailer_thread
        lines.append(f"tailer alive             : {tailer.is_alive() if tailer else None}")
        lines.append(f"log length               : {len(runner.log_widget.value)}")
        lines.append(
            f"progress                 : {runner.progress_widget.value}"
            f"/{runner.progress_widget.max} ({runner.progress_widget.bar_style!r})"
        )
        lines.append(f"history outputs          : {len(runner.history_widget.outputs or ())}")
        lines.append(f"has_phenix               : {runner.has_phenix}")
        if runner.has_phenix:
            lines.append(f"viewer initialized       : {runner._viewer_initialized}")
            lines.append(f"last pdb                 : {runner._last_pdb}")
        import os

        lines.append(f"console.log exists       : {os.path.exists(runner.console_log)}")
        if os.path.exists(runner.console_log):
            lines.append(f"console.log size         : {os.path.getsize(runner.console_log)}")
        lines.append(f"pid file exists          : {os.path.exists(runner.pid_file)}")
    else:
        lines.append("(no runner passed; call report(runner) for the run's state)")

    print("\n".join(lines))


if __name__ == "__main__":
    report()
