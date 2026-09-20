#!/usr/bin/env python
"""Everything worth knowing when the GUI misbehaves on Colab, as data.

Colab's frontend cannot be driven from a terminal, so this is the substitute for
looking at it. `probe()` returns a plain dict and `--json` prints it, because the
point is to diff two of these -- a run that works against a run that does not --
rather than to read prose. `report()` still prints a human block for pasting into
an issue.

Every field is chosen to distinguish a specific failure:

  paths.*    whether the file pickers open on /content, and what the launch-
             directory lookup would have said instead. On Colab it succeeds and
             returns a supervisor's cwd, so a plausible wrong answer used to
             arrive before the /content fallback ever ran.
  packages.* whether the TensorFlow downgrade actually took. Colab ships 2.20,
             abismal pins 2.18, and the install happens under a live kernel: an
             already-imported module keeps the old version in memory, which is a
             successful install that still ends in a broken import.
  gpu.*      whether a GPU runtime actually has a GPU that TensorFlow can see.
  imports.*  torchref in particular, whose absence is otherwise silent -- the
             refinement worker is detached and its stderr goes to a file.

Run it in a cell of the notebook that is misbehaving:

    from devtools.collect_colab_debug import report; report()

or, when devtools is not on the path (the usual case on Colab), fetch it from the
branch under test:

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/rs-station/abismal'
        '/gui/devtools/collect_colab_debug.py'
    ).read())
    report()
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading

SCHEMA = 1

PACKAGES = (
    "abismal",
    "ipywidgets",
    "ipython",
    "jupyterlab",
    "numpy",
    "tensorflow",
    "tensorflow-probability",
    "torch",
    "torchref",
    "reciprocalspaceship",
)

MODULES = ("google.colab", "abismal", "abismal.gui", "tensorflow", "torch", "torchref")


def _safe(fn, default=None):
    """Every probe is best-effort: a broken environment is the thing being reported."""
    try:
        return fn()
    except Exception as error:  # noqa: BLE001 - the error *is* the datum
        return f"<{type(error).__name__}: {error}>" if default is None else default


def _version(name):
    def get():
        import importlib.metadata as md

        return md.version(name)

    return _safe(get)


def _importable(name):
    from importlib.util import find_spec

    # find_spec, not import: torch alongside some TensorFlow versions segfaults,
    # and whether it *could* be imported is the whole question.
    return _safe(lambda: find_spec(name) is not None, default=False)


def _environment():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "on_colab": _importable("google.colab"),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "colab_gpu": os.environ.get("COLAB_GPU"),
        "jpy_parent_pid": os.environ.get("JPY_PARENT_PID"),
        # runner.py imports _is_colab *by value*, so the two can disagree and only
        # the runner's copy decides whether the polling loop was ever set up.
        "is_colab_file_selector": _safe(
            lambda: __import__(
                "abismal.gui.components.file_selector", fromlist=["_is_colab"]
            )._is_colab()
        ),
        "is_colab_runner": _safe(
            lambda: __import__("abismal.gui.runner", fromlist=["_is_colab"])._is_colab()
        ),
    }


def _paths():
    def proc_cwd():
        pid = os.environ.get("JPY_PARENT_PID", "")
        if not pid.isdigit():
            return None
        return os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))

    return {
        "cwd": _safe(os.getcwd),
        "content_exists": os.path.isdir("/content"),
        "proc_parent_cwd": _safe(proc_cwd),
        "pwd_env": os.environ.get("PWD"),
        "jupyter_launch_directory": _safe(
            lambda: __import__(
                "abismal.gui.components.file_selector",
                fromlist=["_jupyter_launch_directory"],
            )._jupyter_launch_directory()
        ),
        "default_directory": _safe(
            lambda: __import__(
                "abismal.gui.components.file_selector", fromlist=["default_directory"]
            ).default_directory()
        ),
    }


def _gpu(tf_preloaded):
    def tf_devices():
        # Only when TensorFlow is already loaded. Importing it here would be
        # invasive: on Colab this probe may run before the install, and importing
        # TF 2.20 first is exactly what makes the downgrade to 2.18 need a restart.
        # A probe must not create the condition it is there to report.
        if "tensorflow" not in sys.modules:
            return None
        import tensorflow as tf

        return [d.name for d in tf.config.list_physical_devices("GPU")]

    def smi():
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() or None

    return {
        "nvidia_smi_present": _safe(
            lambda: __import__("shutil").which("nvidia-smi") is not None, default=False
        ),
        "nvidia_smi": _safe(smi),
        "cuda_version_env": os.environ.get("CUDA_VERSION"),
        # Sampled before the probe ran: probing _is_colab imports abismal.gui,
        # which pulls TensorFlow in, so asking here would always say True and
        # tell us nothing about the kernel we were handed.
        "tensorflow_imported_before_probe": tf_preloaded,
        "tf_gpu_devices": _safe(tf_devices),
    }


def _runner_state(runner):
    tailer = getattr(runner, "_tailer_thread", None)
    state = {
        "pid": _safe(lambda: runner._pid),
        "is_running": _safe(lambda: runner.is_running),
        "monitoring_active": _safe(lambda: runner._monitoring_active),
        "poll_timer_armed": _safe(lambda: runner._poll_timer is not None),
        "tailer_alive": _safe(lambda: tailer.is_alive() if tailer else None),
        "log_length": _safe(lambda: len(runner.log_widget.value)),
        "progress": _safe(
            lambda: f"{runner.progress_widget.value}/{runner.progress_widget.max}"
        ),
        "history_outputs": _safe(lambda: len(runner.history_widget.outputs or ())),
        "has_phenix": _safe(lambda: runner.has_phenix),
        "viewer_initialized": _safe(lambda: runner._viewer_initialized),
        "last_pdb": _safe(lambda: str(runner._last_pdb)),
        "console_log_exists": _safe(lambda: os.path.exists(runner.console_log), False),
        "pid_file_exists": _safe(lambda: os.path.exists(runner.pid_file), False),
    }
    if state["console_log_exists"] is True:
        state["console_log_size"] = _safe(lambda: os.path.getsize(runner.console_log))
    return state


def probe(runner=None):
    """The whole diagnostic as a JSON-serializable dict."""
    tf_preloaded = "tensorflow" in sys.modules
    return {
        "schema": SCHEMA,
        "environment": _environment(),
        "paths": _paths(),
        "packages": {name: _version(name) for name in PACKAGES},
        "imports": {name: _importable(name) for name in MODULES},
        "gpu": _gpu(tf_preloaded),
        "threads": _safe(lambda: sorted(t.name for t in threading.enumerate())),
        "runner": _runner_state(runner) if runner is not None else None,
    }


def _flatten(obj, prefix=""):
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flatten(value, f"{path}.")
        else:
            yield path, value


def report(runner=None):
    """Print the probe as an aligned block, for pasting into an issue."""
    data = probe(runner)
    rows = [(k, v) for k, v in _flatten(data) if not k.startswith("runner.") or runner]
    width = max(len(k) for k in dict(rows))
    print("=== abismal GUI on Colab ===")
    for key, value in rows:
        print(f"{key:<{width}} : {value}")
    if runner is None:
        print("\n(no runner passed; call report(runner) for the run's state)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of an aligned block"
    )
    parser.add_argument("-o", "--out", help="write to this file instead of stdout")
    args = parser.parse_args(argv)

    if args.json:
        text = json.dumps(probe(), indent=2, sort_keys=True, default=str)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
        else:
            print(text)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
