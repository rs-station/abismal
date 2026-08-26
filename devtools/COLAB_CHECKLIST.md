# Colab smoke test

Colab is the one place the automated checks cannot reach: there is no way to drive it
from a development machine. Everything else has been pushed into `tests/gui/` and the
scripts in this directory, so what is left here is deliberately short — run it once
before a release, not on every change.

The *logic* that is Colab-specific is already covered headlessly in
`tests/gui/test_colab.py` (the polling callback and the traits it pushes, the
`_run_on_main_thread` deferral, and a guard that no widget class Colab cannot render
appears in the form). What remains below is only what needs a real Colab frontend.

Ten steps. Each says what you should see, so a deviation is reportable without
interpretation.

| # | Do | Expect |
|---|---|---|
| 1 | Run the install cell | No traceback. It installs `abismal[gui]`; without the extra there is no ipywidgets and step 2 cannot run. |
| 2 | Run the GUI cell | The form appears, with a **row of group buttons** — not tabs. `inputs` and `dmin` at the top, `out_dir` last in that top section. |
| 3 | Click three different group buttons | Exactly one panel visible at a time; the active button is highlighted. |
| 4 | Upload a small mtz to the session, browse to `/content`, add it | The selected-files list updates to name it. |
| 5 | Click Run with `dmin` left empty | argparse usage text appears **in the output area under the button**. This is the failure mode that matters most on Colab: uncaught, the click silently does nothing. |
| 6 | Fill `dmin`, set `--epochs 2`, click Run | The log starts streaming and the progress bar advances **within about 5 s**. |
| 7 | Wait one poll interval | The history plot appears below the progress bar. |
| 7b | On a `--torchref-pdb` run, wait for the first refinement | The 3D viewer shows a model and maps. It used to 404 here because it fetched over `/files/`, which Colab does not serve; the files are embedded in the page now, so this is worth re-checking once. |
| 8 | Scroll up in the log while it is still running | It stays where you put it rather than yanking you back to the bottom; scrolling back to the bottom re-arms the follow. |
| 9 | Click Stop | The button disables and the label changes. |
| 10 | Reload the tab, re-run the GUI cell, click Run | It reconnects to the still-running job rather than starting a second one. |

## Known to be broken, do not report as new

- **Step 10 fails if `console.log` is missing.** Attaching to a job whose log file is
  not there yet kills the tailer thread with an unhandled `FileNotFoundError`, silently,
  because nothing joins a daemon thread. Recorded as an xfail in
  `tests/gui/test_runner_replay.py`.
- **Polling never stops after a refinement run.** `_tail` clears `_monitoring_active`
  only on the no-refinement path, so the browser's `setInterval` keeps running for the
  life of the tab. Harmless but visible in a profiler. Recorded as an xfail in
  `tests/gui/test_colab.py`.
- **The viewer hud reads "Loading..." even when it has finished.** gemmimol leaves it
  there after a successful load; the model and maps are present regardless.

## When something else goes wrong

Add a cell and run:

```python
import urllib.request
exec(urllib.request.urlopen(
    'https://raw.githubusercontent.com/rs-station/abismal/gui/devtools/collect_colab_debug.py'
).read())
report()            # or report(runner) if you have the AbismalRunner handy
```

Paste the output into the issue. It reports which `_is_colab` each module sees — they
are separate, because `runner.py` imports it by value — the installed versions, whether
there is a kernel with an `io_loop` to marshal onto, the live threads, and the runner's
own view of the job: pid, liveness, whether the poll timer is still armed, whether the
tailer is alive, and the sizes of `console.log` and the log widget.

That last pair is usually the answer on its own. A large `console.log` with an empty log
widget means the tailer died; an empty `console.log` means the job never got going.
