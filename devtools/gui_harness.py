"""Drive the abismal notebook GUI from a plain python process.

The GUI is developed against Jupyter and Colab, but almost none of it needs either.
Two properties of the source make that true, and both are load-bearing:

- ``AbismalRunner._run_on_main_thread`` falls through to a synchronous ``fn()`` when
  ``get_ipython()`` returns None, so every widget mutation from the tailer, the poll
  timer and the progress updates lands inline in a plain process.
- ``runner.py`` and ``argparse_gui.py`` contain no ``display()`` calls at all. Every
  output is an assignment to a widget trait, so it can be read back afterwards.

So a headless process can build the form, drive the controls, run a job and read every
byte the GUI would have shown -- including the history plot, which arrives as base64
PNG in ``history_widget.outputs``.

This module is the shared half. ``gui_snapshot.py`` is the command line on top of it,
and ``tests/gui/`` imports the same functions so the two cannot drift.
"""
from __future__ import annotations

import base64
import html as _html
import json
import os
import re
from pathlib import Path

# Set before abismal is imported anywhere. TF's cuFFT/cuDNN registration errors bury
# real output, and matplotlib must not try to find a display.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIR = REPO_ROOT / "tests" / "gui" / "replay"


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_form(parser=None, cls=None):
    """Return a built form.

    ``to_widget()`` must run before ``to_args()`` -- it is what populates
    ``_all_args`` -- and it is not idempotent, so it is called exactly once here.

    ``cls`` defaults to ``ArgparseGUI``, which maps `inputs`, `eff_files` and
    `torchref_pdb` to filesystem-browsing selectors. Pass ``ArgparseGUIBase`` to get
    the plain dispatch instead, which is what a small synthetic parser wants: its
    `inputs` is an ordinary positional, not a reflection-file picker.
    """
    from abismal.gui import ArgparseGUI

    if cls is None:
        cls = ArgparseGUI
    gui = cls(parser=parser) if parser is not None else cls()
    gui.to_widget()
    return gui


def controls(gui):
    """``{dest: widget}`` for every control the form built."""
    return {action.dest: widget for action, widget in gui._all_args.items()}


def set_control(gui, dest, value):
    """Set a control by argparse dest.

    ``_ToggleRow``, ``Text`` and ``Dropdown`` wrap their real widget in an HBox and
    expose ``.value`` as a getter-only property, so assigning to the wrapper raises
    ``AttributeError``. The inner widget is always the last child.
    """
    widget = controls(gui)[dest]

    # The file selectors are VBoxes of a browser UI; their `value` reads a private
    # list of chosen paths rather than any one child widget, so setting the last
    # child would land on a button.
    if hasattr(widget, "_selected_files"):
        widget._selected_files = list(value) if isinstance(value, (list, tuple)) else [value]
        if hasattr(widget, "_update_selected_label"):
            widget._update_selected_label()
        return widget

    target = widget.children[-1] if getattr(widget, "children", None) else widget
    target.value = value
    return target


# ---------------------------------------------------------------------------
# Replaying a run
# ---------------------------------------------------------------------------

def make_results_template(directory):
    """Write a tiny refined.pdb + refined.mtz for the stub to copy per epoch.

    Generated rather than committed: the real per-epoch outputs are 1-4 MB each, and
    generating them keeps the column schema visible here instead of hidden inside a
    binary. The columns are torchref's spelling, which is what GemmiMolViewer.map_keys
    now looks for.
    """
    import numpy as np
    import reciprocalspaceship as rs

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    directory.joinpath("refined.pdb").write_text(
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       6.500   5.200   5.100  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1       7.100   6.500   5.600  1.00 20.00           C\n"
        "ATOM      4  O   ALA A   1       8.300   6.700   5.500  1.00 20.00           O\n"
        "END\n"
    )

    hkl = [(h, k, l) for h in range(1, 4) for k in range(1, 4) for l in range(1, 4)]
    rng = np.random.default_rng(0)
    n = len(hkl)
    ds = rs.DataSet(
        {
            "H": np.array([i[0] for i in hkl], "int32"),
            "K": np.array([i[1] for i in hkl], "int32"),
            "L": np.array([i[2] for i in hkl], "int32"),
            "FWT": rng.uniform(10, 100, n).astype("float32"),
            "PHWT": rng.uniform(-180, 180, n).astype("float32"),
            "ANOM": rng.uniform(0, 5, n).astype("float32"),
            "PANOM": rng.uniform(-180, 180, n).astype("float32"),
        },
        cell=[30.0, 30.0, 30.0, 90.0, 90.0, 90.0],
        spacegroup="P 1",
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)
    ds.write_mtz(str(directory / "refined.mtz"))
    return directory


def stub_env(out_dir, *, console=None, history=None, results=None, delay=0.0,
             prefix="torchref", exit_code=0, hang=False, ignore_sigterm=False):
    """Environment for the replay stub. See tests/gui/replay/abismal."""
    env = {
        "ABISMAL_REPLAY_CONSOLE": str(console or REPLAY_DIR / "console.log"),
        "ABISMAL_REPLAY_OUTDIR": str(out_dir),
        "ABISMAL_REPLAY_PREFIX": prefix,
        "ABISMAL_REPLAY_DELAY": str(delay),
        "ABISMAL_REPLAY_EXIT": str(exit_code),
    }
    if history is not False:
        env["ABISMAL_REPLAY_HISTORY"] = str(history or REPLAY_DIR / "history.csv")
    if results:
        env["ABISMAL_REPLAY_RESULTS"] = str(results)
    if hang:
        env["ABISMAL_REPLAY_HANG"] = "1"
    if ignore_sigterm:
        env["ABISMAL_REPLAY_IGNORE_SIGTERM"] = "1"
    return env


def start_replay(out_dir, *, has_phenix=True, total_epochs=12, poll_interval=0.05,
                 args=(), **stub_kwargs):
    """Start a replay through the real ``AbismalRunner.start()``.

    Puts the stub first on PATH rather than patching Popen, so the launch, the pid
    file, ``/proc`` liveness and the exit code all stay real. Returns the runner
    without waiting -- use :func:`wait_for_replay`.
    """
    from abismal.gui.runner import AbismalRunner

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PATH"] = f"{REPLAY_DIR}{os.pathsep}{os.environ['PATH']}"
    os.environ.update(stub_env(out_dir, **stub_kwargs))

    AbismalRunner.poll_interval = poll_interval
    runner = AbismalRunner(
        args=list(args), out_dir=str(out_dir),
        has_phenix=has_phenix, total_epochs=total_epochs,
    )
    runner.start()
    return runner


def wait_for_replay(runner, timeout=60.0):
    """Block until the tailer thread has drained and finished."""
    thread = runner._tailer_thread
    if thread is not None:
        thread.join(timeout)
    return runner


def quiesce(runner, timeout=5.0):
    """Stop a runner and its threads. Safe to call more than once.

    Nothing in AbismalRunner cancels the self-rescheduling poll Timer, and with
    has_phenix=True ``_monitoring_active`` is never cleared, so a runner left alone
    keeps a timer chain and a watcher thread alive. Tests and the CLI both need a way
    to put one down.
    """
    import os as _os
    import signal as _signal

    runner._monitoring_active = False
    timer = getattr(runner, "_poll_timer", None)
    if timer is not None:
        timer.cancel()
    pid = getattr(runner, "_pid", None)
    if pid and runner.is_running:
        try:
            _os.killpg(_os.getpgid(pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                _os.kill(pid, _signal.SIGKILL)
            except OSError:
                pass
    thread = getattr(runner, "_tailer_thread", None)
    if thread is not None:
        thread.join(timeout)
    return runner


# ---------------------------------------------------------------------------
# Reading the tree
# ---------------------------------------------------------------------------

# Per-class trait whitelist. `Widget.get_state()` works headlessly but is dominated by
# model ids and defaults, which makes it useless to diff; this keeps a snapshot to the
# traits a human would actually look at.
_TRAITS = {
    "HTML": ("value",),
    "Label": ("value",),
    "Text": ("value", "placeholder", "disabled"),
    "Textarea": ("value", "placeholder"),
    "Dropdown": ("value", "options", "disabled"),
    "Select": ("value", "options"),
    "SelectMultiple": ("value", "options"),
    "Button": ("description", "disabled", "button_style"),
    "ToggleButton": ("description", "value", "button_style"),
    "Checkbox": ("description", "value"),
    "IntProgress": ("value", "min", "max", "bar_style"),
    "IntText": ("value",),
    "FloatText": ("value",),
}
# Only the layout keys that actually drive this GUI's layout. `display` is how
# _make_group_container shows one panel at a time, so it must be visible in a snapshot.
_LAYOUT_KEYS = ("display", "width", "min_width", "height", "flex", "overflow_y", "border")

_MAX_VALUE = 120


def _short(value):
    text = repr(value)
    if len(text) > _MAX_VALUE:
        text = text[: _MAX_VALUE - 3] + "..."
    return text


def _outputs_summary(widget):
    """Describe an Output widget's payload by mime type and size, not by content."""
    out = []
    for item in getattr(widget, "outputs", ()) or ():
        data = item.get("data", {})
        for mime, payload in data.items():
            if mime == "image/png":
                out.append(f"{mime}({len(base64.b64decode(payload))}B)")
            else:
                out.append(f"{mime}({len(payload)}ch)")
    return out


def tree_data(widget, path="root"):
    """Recursive, JSON-able description of a widget tree."""
    cls = type(widget).__name__
    node = {"class": cls, "path": path}

    for trait in _TRAITS.get(cls, ()):
        if widget.has_trait(trait) if hasattr(widget, "has_trait") else hasattr(widget, trait):
            node[trait] = getattr(widget, trait)

    layout = getattr(widget, "layout", None)
    if layout is not None:
        lay = {}
        for key in _LAYOUT_KEYS:
            value = getattr(layout, key, None)
            if value is None:
                continue
            # `display` is kept even when empty: _make_group_container shows a panel by
            # setting display='' and hides it with 'none', so an empty string is the
            # signal that this is the visible group. Dropping it would make "visible"
            # indistinguishable from "never set".
            if value == "" and key != "display":
                continue
            lay[key] = "<shown>" if (key == "display" and value == "") else value
        if lay:
            node["layout"] = lay

    if cls == "Output":
        summary = _outputs_summary(widget)
        if summary:
            node["outputs"] = summary

    children = getattr(widget, "children", None)
    if children:
        node["children"] = [
            tree_data(child, f"{path}/{i}:{type(child).__name__}")
            for i, child in enumerate(children)
        ]
    return node


def dump_tree(widget):
    """Indented one-line-per-widget rendering of :func:`tree_data`."""
    lines = []

    def walk(node, depth):
        cls = node["class"]
        bits = []
        for key, value in node.items():
            if key in ("class", "path", "children", "layout", "outputs"):
                continue
            bits.append(f"{key}={_short(value)}")
        if node.get("outputs"):
            bits.append("outputs=[" + ", ".join(node["outputs"]) + "]")
        if node.get("layout"):
            lay = ",".join(f"{k}={v}" for k, v in node["layout"].items())
            bits.append(f"layout({lay})")
        kids = node.get("children")
        head = f"{cls}" + (f" [{len(kids)} children]" if kids else "")
        lines.append("  " * depth + head + ("   " + "  ".join(bits) if bits else ""))
        for kid in kids or ():
            walk(kid, depth + 1)

    walk(tree_data(widget), 0)
    return "\n".join(lines) + "\n"


def iter_widgets(widget):
    yield widget
    for child in getattr(widget, "children", None) or ():
        yield from iter_widgets(child)


# ---------------------------------------------------------------------------
# Extracting payloads
# ---------------------------------------------------------------------------

def extract_pngs(widget):
    """``[(path, bytes), ...]`` for every image/png in the tree."""
    found = []
    for node in iter_widgets(widget):
        for i, item in enumerate(getattr(node, "outputs", ()) or ()):
            payload = item.get("data", {}).get("image/png")
            if payload:
                found.append((f"{type(node).__name__}_{i}", base64.b64decode(payload)))
    return found


def extract_scripts(widget):
    """``[(name, source), ...]`` for every application/javascript payload."""
    found = []
    for node in iter_widgets(widget):
        for i, item in enumerate(getattr(node, "outputs", ()) or ()):
            payload = item.get("data", {}).get("application/javascript")
            if payload:
                found.append((f"{type(node).__name__}_{i}", payload))
    return found


_PRE = re.compile(r"</?pre[^>]*>")


def log_text(runner):
    """The log widget's contents as plain text.

    ``_render_log_html`` escapes and wraps in ``<pre>``; undo both so the streamed log
    reads as it would in a terminal.
    """
    return _html.unescape(_PRE.sub("", runner.log_widget.value))


# ---------------------------------------------------------------------------
# Writing artifacts
# ---------------------------------------------------------------------------

def write_artifacts(out_dir, *, widget=None, runner=None, gui=None, extra=None):
    """Write everything readable about a built GUI into ``out_dir``.

    Returns a ``{name: path}`` map of what was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    def put(name, text, mode="w"):
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "wb":
            path.write_bytes(text)
        else:
            path.write_text(text)
        written[name] = path

    if widget is not None:
        put("tree.txt", dump_tree(widget))
        put("tree.json", json.dumps(tree_data(widget), indent=2, default=str))
        for name, png in extract_pngs(widget):
            put(f"outputs/{name}.png", png, mode="wb")
        for name, js in extract_scripts(widget):
            put(f"js/{name}.js", js)

    if gui is not None:
        try:
            put("argv.txt", " ".join(gui.to_args()) + "\n")
        except Exception as error:  # a half-filled form is a normal state to snapshot
            put("argv.txt", f"<to_args() raised {type(error).__name__}: {error}>\n")

    if runner is not None:
        put("log.txt", log_text(runner))
        for name, png in extract_pngs(runner.history_widget):
            put("history.png" if name.startswith("Output") else f"{name}.png", png, mode="wb")

    for name, text in (extra or {}).items():
        put(name, text)

    return written
