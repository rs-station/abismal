#!/usr/bin/env python
"""The sealed viewer must render, reload, and leave its host page alone.

    python devtools/browser/check_viewer_isolation.py

The viewer's stylesheet is written for a page it owns: a global `*` reset plus
`background-color: black; height: 600px; overflow: hidden` on html and body.
Jupyter's `isolated: True` metadata is supposed to keep that in its own frame.
JupyterLab honours it; Colab does not, and the rules landed on the notebook --
everything below the viewer went black and clipped, which looked like the GUI
vanishing rather than a stylesheet escaping.

So the viewer is sealed in an iframe via srcdoc, and this checks that in a real
browser: the host keeps its background, its overflow and the content beneath,
while the viewer still loads a model and still answers a reload broadcast.

Swap `iframe_html` for `html` below and it fails on overflow -- that is the bug
reproducing, and worth re-running if this ever starts passing vacuously.

Needs:
    pip install -r devtools/requirements-browser.txt
    playwright install chromium
"""
import sys
from pathlib import Path
sys.path.insert(0, "devtools")
import gui_harness as H
from playwright.sync_api import sync_playwright
from abismal.gui.components.gemmimol import GemmiMolViewer

out = Path("/tmp/iframe-check"); out.mkdir(parents=True, exist_ok=True)
template = H.make_results_template(out / "_t")
viewer = GemmiMolViewer(pdb_file=str(template / "refined.pdb"),
                        mtz_file=str(template / "refined.mtz"),
                        viewer_id="sealed")

# A host page that looks like a notebook: white, tall, with content below.
host = out / "host.html"
host.write_text(
    "<body style='background:#fff'>"
    "<div id='above'>ABOVE</div>"
    + viewer.iframe_html +
    "<div id='below' style='height:400px'>BELOW</div></body>"
)

problems = []
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1000, "height": 900})
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.goto(host.as_uri(), timeout=60000)

    # the host page must keep its own styling
    bg = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    below_visible = page.evaluate(
        "() => { const el = document.getElementById('below');"
        "        return el && el.getBoundingClientRect().height > 0; }")
    host_overflow = page.evaluate("() => getComputedStyle(document.documentElement).overflow")

    frame = page.frames[1] if len(page.frames) > 1 else None
    if frame is None:
        problems.append("no iframe was created")
    else:
        frame.wait_for_function(
            "() => window.V && (V.model_bags || []).length >= 1", timeout=60000)
        loaded = frame.evaluate("() => (V.model_bags || []).length")
        vid = frame.evaluate("() => window.ABISMAL_VIEWER_ID")
        if loaded < 1: problems.append("model did not load inside the iframe")
        if vid != "sealed": problems.append(f"viewer id is {vid!r}")

        # broadcast reload, as runner.py does
        frame.evaluate("""() => { window.__reloads = 0;
            const o = V.load_model.bind(V);
            V.load_model = function(){ window.__reloads++; return o.apply(V, arguments); }; }""")
        page.evaluate("(p) => { document.querySelectorAll('iframe').forEach("
                      "f => f.contentWindow.postMessage(p, '*')); }", viewer.reload_payload)
        page.wait_for_timeout(4000)
        if frame.evaluate("() => window.__reloads") < 1:
            problems.append("the sealed viewer ignored a reload broadcast")

    page.screenshot(path=str(out / "sealed.png"), full_page=True)
    b.close()

print("host background   :", bg, "(must stay white)")
print("content below     :", "visible" if below_visible else "MISSING")
print("host overflow     :", host_overflow, "(must not be hidden)")
if bg != "rgb(255, 255, 255)": problems.append(f"host background became {bg}")
if not below_visible: problems.append("content below the viewer was clipped")
if host_overflow == "hidden": problems.append("viewer clamped the host's overflow")
print()
print("FAILED:" if problems else "OK")
for p in problems: print("  -", p)
