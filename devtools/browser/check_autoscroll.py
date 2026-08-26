#!/usr/bin/env python
"""Check the log box's sticky-bottom autoscroll in a real browser.

    python devtools/browser/check_autoscroll.py [--out-dir DIR]

The autoscroll is a MutationObserver that keeps the log pinned to the bottom while new
lines arrive, but stops doing so the moment the reader scrolls up -- otherwise reading
back through a long log is impossible during a run. None of that can be checked without
a layout engine: it is entirely scrollTop, clientHeight and scrollHeight.

No Jupyter is involved. The page is a bare div with the same `.abismal-log-scroll`
class and geometry the runner gives its log box, and **the javascript is read out of
`AbismalRunner._log_js_widget`** rather than copied here, so the check cannot drift from
what ships.

Local only. See devtools/requirements-browser.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gui_harness as H  # noqa: E402

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body {{ margin: 0; font-family: sans-serif; }}
  .abismal-log-scroll {{
      height: 300px; overflow-y: auto; border: 1px solid #ccc; width: 600px;
  }}
  pre {{ margin: 0; font-family: monospace; font-size: 12px;
         white-space: pre-wrap; line-height: 1.3; }}
</style></head>
<body>
  <div class="abismal-log-scroll" id="box"><pre id="log">{log}</pre></div>
  <script>{js}</script>
  <script>
    // Stand-in for _append_log: the runner replaces the whole <pre> contents, which is
    // what the observer is watching for.
    window.appendLines = function(n) {{
      var pre = document.getElementById('log');
      var extra = '';
      for (var i = 0; i < n; i++) {{ extra += 'appended line ' + i + '\\n'; }}
      pre.textContent = pre.textContent + extra;
    }};
    window.boxState = function() {{
      var b = document.getElementById('box');
      return {{scrollTop: b.scrollTop, clientHeight: b.clientHeight,
               scrollHeight: b.scrollHeight,
               atBottom: (b.scrollTop + b.clientHeight) >= (b.scrollHeight - 5)}};
    }};
  </script>
</body></html>
"""


def build_page(out_dir):
    """Write the harness page, taking the javascript from the runner itself."""
    from abismal.gui.runner import AbismalRunner

    runner = AbismalRunner(args=None, out_dir=str(out_dir), has_phenix=False)
    scripts = H.extract_scripts(runner._log_js_widget)
    if not scripts:
        raise SystemExit("the runner emitted no autoscroll javascript")
    js = scripts[0][1]
    runner.shutdown()

    log = "".join(f"initial line {i}\n" for i in range(80))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "autoscroll.html"
    page.write_text(PAGE.format(js=js, log=log))
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default="/tmp/abismal-autoscroll-check")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "playwright is not installed.\n"
            "  pip install -r devtools/requirements-browser.txt\n"
            "  playwright install chromium"
        )

    out_dir = Path(args.out_dir)
    page_path = build_page(out_dir)
    problems = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 800, "height": 600})
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(page_path.as_uri())

        # The script polls every 100 ms for the box before attaching.
        page.wait_for_function("() => document.getElementById('box').__abismal_autoscroll",
                               timeout=10_000)

        # 1. new output keeps the view pinned to the bottom
        page.evaluate("() => window.appendLines(40)")
        page.wait_for_timeout(300)
        state = page.evaluate("() => window.boxState()")
        if not state["atBottom"]:
            problems.append(f"did not stick to the bottom: {state}")
        page.screenshot(path=str(out_dir / "autoscroll_bottom.png"))

        # 2. scrolling up releases it, so a reader can look back mid-run
        page.evaluate("() => { document.getElementById('box').scrollTop = 0; }")
        page.wait_for_timeout(150)
        page.evaluate("() => window.appendLines(40)")
        page.wait_for_timeout(300)
        scrolled = page.evaluate("() => window.boxState()")
        if scrolled["scrollTop"] > 50:
            problems.append(
                f"jumped back to the bottom while scrolled up: {scrolled}"
            )
        page.screenshot(path=str(out_dir / "autoscroll_held.png"))

        # 3. returning to the bottom re-arms it
        page.evaluate(
            "() => { const b = document.getElementById('box');"
            " b.scrollTop = b.scrollHeight; }"
        )
        page.wait_for_timeout(150)
        page.evaluate("() => window.appendLines(40)")
        page.wait_for_timeout(300)
        rearmed = page.evaluate("() => window.boxState()")
        if not rearmed["atBottom"]:
            problems.append(f"did not re-arm after returning to the bottom: {rearmed}")
        page.screenshot(path=str(out_dir / "autoscroll_rearmed.png"))

        browser.close()

    print(f"  page:        {page_path}")
    print(f"  pinned:      {state}")
    print(f"  held back:   {scrolled}")
    print(f"  re-armed:    {rearmed}")
    print(f"  screenshots: {out_dir}/autoscroll_*.png")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
