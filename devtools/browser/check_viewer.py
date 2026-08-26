#!/usr/bin/env python
"""Load the 3D viewer in a real browser and check it actually renders.

    python devtools/browser/check_viewer.py [--pdb X --mtz Y] [--out-dir DIR]

This is the one part of the GUI that cannot be checked headlessly: whether 3Dmol and
gemmi.js load, parse the files and put something on screen. Everything else about the
viewer -- the column selection, the template substitution, the reload payload -- is
covered by tests/gui/test_gemmimol.py.

No Jupyter is involved. `GemmiMolViewer.html` is a complete standalone document, so it
is served straight from a temporary directory. It has to be served over HTTP rather
than opened as a file:// URL, because the viewer fetches the pdb and mtz with XHR and
the file:// origin is opaque, so every fetch is blocked.

Local only. Not in pyproject.toml, not in CI, and outside pytest's testpaths so it
cannot be collected by accident. Needs:

    pip install -r devtools/requirements-browser.txt
    playwright install chromium
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import shutil
import socket
import socketserver
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gui_harness as H  # noqa: E402


@contextlib.contextmanager
def serve(directory):
    """A background HTTP server rooted at `directory`, on a free port."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

        def handle_error(self, request, client_address):
            pass  # a viewer that gives up mid-fetch is not a server error

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = Quiet(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def stage(out_dir, pdb, mtz):
    """Write viewer.html next to copies of the pdb/mtz, with bare relative URLs."""
    from abismal.gui.components.gemmimol import GemmiMolViewer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdb, out_dir / "model.pdb")
    shutil.copy(mtz, out_dir / "data.mtz")

    viewer = GemmiMolViewer(
        pdb_file=str(out_dir / "model.pdb"),
        mtz_file=str(out_dir / "data.mtz"),
        pdb_url="model.pdb",
        mtz_url="data.mtz",
        viewer_id="check-viewer",
    )
    (out_dir / "viewer.html").write_text(viewer.html)
    return viewer.map_keys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdb")
    parser.add_argument("--mtz")
    parser.add_argument("--out-dir", default="/tmp/abismal-viewer-check")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--headed", action="store_true", help="watch it run")
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
    if args.pdb and args.mtz:
        pdb, mtz = Path(args.pdb), Path(args.mtz)
    else:
        template = H.make_results_template(out_dir / "_template")
        pdb, mtz = template / "refined.pdb", template / "refined.mtz"
        print(f"  using the generated template model at {template}")

    map_keys = stage(out_dir, pdb, mtz)
    print(f"  map_keys: {map_keys}")

    problems = []
    with serve(out_dir) as base_url, sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1100, "height": 800})

        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on(
            "requestfailed",
            lambda r: problems.append(f"failed request: {r.url} ({r.failure})"),
        )

        page.goto(f"{base_url}/viewer.html", timeout=args.timeout * 1000)

        # Wait on the viewer's own state rather than the hud text. The hud narrates
        # "Loading PDB..." then "Loading MTZ...", but gemmimol leaves it reading
        # "Loading..." after a successful load, so the text is not a completion signal.
        # model_bags and map_bags are: they are what the reload path itself clears and
        # refills.
        expected_maps = len(map_keys) // 2

        def wait_loaded(what):
            try:
                page.wait_for_function(
                    "(n) => window.V && (V.model_bags || []).length >= 1"
                    " && (V.map_bags || []).length >= n",
                    arg=expected_maps,
                    timeout=args.timeout * 1000,
                )
            except Exception:
                problems.append(f"the viewer never finished {what}")

        wait_loaded("loading")

        hud = page.evaluate(
            "() => (document.getElementById('hud') || {}).textContent || ''"
        ).strip()
        if hud.startswith("Error:"):
            problems.append(f"hud reports {hud!r}")

        models = page.evaluate("() => (window.V && V.model_bags || []).length")
        maps = page.evaluate("() => (window.V && V.map_bags || []).length")
        viewer_id = page.evaluate("() => window.ABISMAL_VIEWER_ID")

        if models < 1:
            problems.append(f"no model loaded (model_bags={models})")
        if maps < 1:
            problems.append(f"no maps loaded (map_bags={maps}, map_keys={map_keys})")
        if viewer_id != "check-viewer":
            problems.append(f"viewer id is {viewer_id!r}")

        page.screenshot(path=str(out_dir / "viewer.png"))

        # The reload path, without Jupyter: this is what runner._render_epoch's
        # postMessage payload does for every epoch after the first, and re-embedding
        # instead would reset the camera.
        # gemmimol's camera is not necessarily a THREE.Vector3, so read the position
        # in whatever shape it comes in rather than assuming toArray().
        read_camera = """() => {
            try {
                const p = window.V && V.camera && V.camera.position;
                if (!p) return null;
                if (typeof p.toArray === 'function') return p.toArray();
                if ('x' in p) return [p.x, p.y, p.z];
                return Array.from(p);
            } catch (e) { return null; }
        }"""
        before = page.evaluate(read_camera)
        page.evaluate(
            """([pdb, mtz, keys]) => window.postMessage(
                   {type: 'reload', pdb_file: pdb, mtz_file: mtz, map_keys: keys}, '*')""",
            ["model.pdb", "data.mtz", map_keys],
        )
        wait_loaded("reloading")

        after = page.evaluate(read_camera)
        if page.evaluate("() => (window.V && V.model_bags || []).length") < 1:
            problems.append("the model did not come back after the reload")
        if before and after and before != after:
            problems.append(f"the camera moved on reload: {before} -> {after}")

        page.screenshot(path=str(out_dir / "viewer_reloaded.png"))
        browser.close()

    (out_dir / "viewer_console.txt").write_text("\n".join(console) + "\n")

    print(f"  hud: {hud!r}")
    print(f"  model_bags={models}  map_bags={maps}")
    print(f"  screenshots: {out_dir/'viewer.png'}, {out_dir/'viewer_reloaded.png'}")
    print(f"  console: {out_dir/'viewer_console.txt'} ({len(console)} lines)")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
