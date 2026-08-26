# devtools

Development tooling for the notebook GUI. Nothing here is packaged: `pyproject.toml`
names `abismal*` explicitly, and this directory sits outside pytest's `testpaths`, so
CI cannot collect it by accident.

It exists so that developing `abismal/gui` does not require a person to restart a
kernel, launch a merge job, and describe what appeared on screen. Almost none of the
GUI needs a frontend to exercise: ipywidgets falls back to a no-op comm with no kernel
attached, `AbismalRunner._run_on_main_thread` applies its closure synchronously when
there is no `get_ipython()`, and neither `runner.py` nor `argparse_gui.py` calls
`display()` — every output is a widget trait that can be read straight back.

## The three commands

```bash
# 1. The form, as text. 2.5 s, no filesystem, no subprocess.
python devtools/gui_snapshot.py --scenario form --out-dir /tmp/snap

# 2. A whole 12-epoch run: launch, log tailing, progress, history plot, per-epoch
#    results. 3.5 s, no GPU, no merge job.
python devtools/gui_snapshot.py --scenario replay --out-dir /tmp/snap

# 3. An existing output directory, read back as the GUI would show it.
python devtools/gui_snapshot.py --scenario runner --out-dir /tmp/snap --results RUNDIR
```

Each writes into `--out-dir`:

| file | what it is |
| --- | --- |
| `summary.md` | read this first: timings, and the handful of facts worth asserting |
| `history.png` | the real plot, decoded out of `history_widget.outputs` — **open it** |
| `tree.txt` | the whole widget tree, one line per widget, with the traits that matter |
| `tree.json` | the same, machine-readable |
| `log.txt` | the streamed log, with the `<pre>` and HTML entities undone |
| `argv.txt` | the command line the form would run |
| `viewer.html` | the standalone 3D viewer document |
| `js/*.js` | the autoscroll, viewer-reload and Colab-poll payloads |

`tree.txt` is a golden-diffable rendering. `display=<shown>` marks the one visible
group panel — `_make_group_container` shows a panel by setting `display` to the empty
string, so an empty value is meaningful and is reported rather than dropped.

## Tests

```bash
pytest tests/gui -p no:xdist -o addopts=""      # ~4 s
```

`tests/gui/` imports `gui_harness` from here, wired up by `pythonpath = ["devtools"]`
in `pyproject.toml`, so the tests and the snapshot CLI cannot drift.

`tests/gui/replay/abismal` is a stdlib-only stand-in for the abismal console script.
`AbismalRunner.start()` resolves a bare `abismal` from `PATH` and passes
`os.environ.copy()` to `Popen`, so putting that directory first on `PATH` substitutes
the executable and nothing else: the real `Popen` call, `start_new_session`, the stdout
redirect, the pid file, `/proc` liveness and the exit code all stay live. That matters
because `_pid_is_abismal` drives `is_running`, `_tail`'s exit condition and the pid-file
lifecycle — mocking `Popen` would force mocking that too, and most of what the tests
cover would become stub talking to stub.

It skips itself when ipywidgets is absent, unless `ABISMAL_REQUIRE_GUI_TESTS` is set,
which CI sets — so the suite cannot silently shrink to nothing.

## Browser checks — local only

Three behaviours need a layout engine and a javascript runtime. Two are covered:

```bash
pip install -r devtools/requirements-browser.txt
playwright install chromium        # ~150 MB, once

python devtools/browser/check_viewer.py       # 3D viewer: renders, and reloads in place
python devtools/browser/check_autoscroll.py   # log box sticky-bottom behaviour
```

Both write screenshots. Neither needs Jupyter: `GemmiMolViewer.html` is a standalone
document, and the autoscroll javascript only looks for a `.abismal-log-scroll` element.
`check_autoscroll.py` reads that javascript out of `AbismalRunner._log_js_widget` rather
than copying it, so it cannot drift from what ships.

`check_viewer.py` serves over HTTP rather than `file://` — the viewer fetches the pdb
and mtz by XHR, and a `file://` origin is opaque, so every fetch would be blocked.

The third, Colab's polling loop, cannot be reached from here at all. Its *logic* is
covered in `tests/gui/test_colab.py` against a fake `google.colab`; what is left is
`COLAB_CHECKLIST.md`, ten steps to run once before a release, and
`collect_colab_debug.py` to paste into an issue when something there goes wrong.
