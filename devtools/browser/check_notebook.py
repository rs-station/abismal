#!/usr/bin/env python
"""Render the GUI in a real browser, on Colab's exact Python stack.

    python devtools/colab_container.py serve        # in another terminal
    python devtools/browser/check_notebook.py

This is the local half of Colab debugging. The container's nbclassic UI is a real
browser frontend with a real widget manager, running the same interpreter and the
same package versions Colab ships -- Python 3.12.13, ipywidgets 8.1.9, TF 2.20 --
so everything except Colab's own proprietary frontend can be checked here, with
no Google round trip and nobody clicking anything.

What it cannot answer, and nothing local can: whether *Colab's* widget manager
renders ipywidgets 8, whether *Colab* executes `application/javascript` from a
widget's `.outputs`, and whether *Colab* honours `isolated: True`. Those three
need a real Colab tab; see devtools/colab_deploy.py.

The notebook it drives imports the live checkout from the bind mount rather than
the installed copy, so an edit is visible on the next run with no reinstall.

Needs:
    pip install -r devtools/requirements-browser.txt
    playwright install chromium
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CONTAINER = "abismal-colab-serve"
MOUNT = "/content/abismal"
DRIVER = "/content/_check_notebook.ipynb"

# nbclassic asks for the legacy notebook-extension shims on every page load and
# gets HTML 404s back. ipywidgets 8 renders through a different path entirely --
# 400-odd widgets appear regardless -- so these are noise from a frontend we are
# only borrowing, not a defect in anything of ours. Matched narrowly: any other
# failed request is still a failure.
BENIGN_REQUESTS = re.compile(r"/nbextensions/(ipyevents|widgets)/")

CELL = f'''import sys, os
# The live checkout ahead of the installed copy: this exists to test the working
# tree, and a reinstall per edit is what makes a debug loop slow.
sys.path.insert(0, "{MOUNT}")
# The replay stub, so a run launches a real subprocess that finishes in seconds.
os.environ["PATH"] = "{MOUNT}/tests/gui/replay:" + os.environ["PATH"]

import abismal, ipywidgets
from abismal.gui import ArgparseGUI
print("abismal:", abismal.__file__)
print("ipywidgets:", ipywidgets.__version__)

gui = ArgparseGUI()
gui.to_widget()'''


def podman(*args, **kwargs):
    return subprocess.run(["podman", *args], capture_output=True, text=True, **kwargs)


def container_token():
    """The running container's jupyter token, read from the server itself."""
    got = podman("exec", CONTAINER, "bash", "-lc",
                 "jupyter server list 2>/dev/null || jupyter notebook list 2>/dev/null")
    if got.returncode != 0:
        raise SystemExit(
            f"container {CONTAINER!r} is not running.\n"
            "  python devtools/colab_container.py serve"
        )
    found = re.search(r"token=([0-9a-f]+)", got.stdout)
    if not found:
        raise SystemExit(f"no token in:\n{got.stdout}\n{got.stderr}")
    return found.group(1)


def install_driver():
    """Write the driver notebook inside the container, not into the checkout."""
    notebook = {
        "cells": [{
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": CELL.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    got = subprocess.run(
        ["podman", "exec", "-i", CONTAINER, "bash", "-c", f"cat > {DRIVER}"],
        input=json.dumps(notebook, indent=1), capture_output=True, text=True,
    )
    if got.returncode != 0:
        raise SystemExit(f"could not write {DRIVER}: {got.stderr}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=8890,
                        help="host port for the container's nbclassic UI")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--headed", action="store_true", help="watch it run")
    parser.add_argument("--out-dir", default="devtools/artifacts/colab")
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "playwright is not installed.\n"
            "  pip install -r devtools/requirements-browser.txt\n"
            "  playwright install chromium"
        )

    token = container_token()
    install_driver()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/notebooks{DRIVER}?token={token}"

    problems, console = [], []
    ms = args.timeout * 1000

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("requestfailed", lambda r: (
            None if BENIGN_REQUESTS.search(r.url)
            else problems.append(f"failed request: {r.url} ({r.failure})")
        ))

        page.goto(url, timeout=ms)
        page.wait_for_function("() => window.Jupyter && Jupyter.notebook", timeout=ms)
        page.wait_for_function(
            "() => Jupyter.notebook.kernel && Jupyter.notebook.kernel.is_connected()",
            timeout=ms)
        # Restart first. The container's kernel outlives a page load, so without
        # this a run inherits whatever the last one imported -- and since the
        # notebook deliberately imports from a mount that changes between runs,
        # a stale module is exactly the thing that would go unnoticed.
        page.evaluate("() => Jupyter.notebook.kernel.restart()")
        page.wait_for_function(
            "() => Jupyter.notebook.kernel && Jupyter.notebook.kernel.is_connected()",
            timeout=ms)
        page.wait_for_timeout(2000)
        page.evaluate("() => Jupyter.notebook.execute_all_cells()")

        # input_prompt_number is '*' while running and a number when done. Waiting
        # on the widget DOM instead would pass as soon as the first child mounts,
        # which is not the same as the form having finished building.
        try:
            page.wait_for_function(
                "() => { const n = Jupyter.notebook.get_cell(0).input_prompt_number;"
                "        return n !== '*' && n !== null && n !== undefined; }",
                timeout=ms)
        except Exception:
            problems.append("the cell never finished executing")

        page.wait_for_timeout(5000)  # let the widget views mount

        state = page.evaluate("""() => {
            const widgets = document.querySelectorAll('.jupyter-widgets');
            const text = Array.from(document.querySelectorAll('.output_area'))
                              .map(o => o.innerText).join('\\n');
            return {
                widgets: widgets.length,
                text,
                buttons: Array.from(document.querySelectorAll('.jupyter-button'))
                              .map(b => b.textContent.trim()).filter(Boolean),
            };
        }""")

        page.screenshot(path=str(out_dir / "notebook.png"), full_page=True)
        browser.close()

    # A widget manager that fails silently renders nothing while erroring nowhere,
    # so the count is the assertion that catches it.
    if state["widgets"] < 50:
        problems.append(
            f"only {state['widgets']} widget elements rendered; the form did not build"
        )
    if "ipywidgets: 8" not in state["text"]:
        problems.append("the kernel did not report ipywidgets 8")
    if f"abismal: {MOUNT}/" not in state["text"]:
        problems.append("the notebook imported an installed abismal, not the mount")

    # The file pickers must open on /content here: google.colab is importable in
    # this image, so this is the same code path Colab takes.
    if "/content" not in state["text"]:
        problems.append("the file selector did not open on /content")

    for expected in ("Add Selected Files", "Run Abismal"):
        if not any(expected in b for b in state["buttons"]):
            problems.append(f"no {expected!r} button rendered")

    (out_dir / "notebook_console.txt").write_text("\n".join(console) + "\n")
    print(f"  widgets rendered : {state['widgets']}")
    print(f"  buttons          : {len(state['buttons'])}")
    print(f"  screenshot       : {out_dir / 'notebook.png'}")
    print(f"  console          : {out_dir / 'notebook_console.txt'} ({len(console)} lines)")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
