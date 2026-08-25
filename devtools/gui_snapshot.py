#!/usr/bin/env python
"""Dump the abismal notebook GUI to files you can read without a browser.

    python devtools/gui_snapshot.py --scenario form   --out-dir /tmp/snap
    python devtools/gui_snapshot.py --scenario replay --out-dir /tmp/snap
    python devtools/gui_snapshot.py --scenario runner --out-dir /tmp/snap --results DIR
    python devtools/gui_snapshot.py --scenario viewer --out-dir /tmp/snap --pdb X --mtz Y

`replay` launches a stub abismal through the real subprocess path and snapshots a
complete run in a couple of seconds. `runner` reads an existing output directory.

Writes `tree.txt` (the whole widget tree, one line per widget), `tree.json`,
`history.png` (the real plot, decoded out of the widget), `log.txt`, `argv.txt`, any
javascript payloads, and `summary.md` as the index. Reading those replaces asking
someone to describe what they see on screen.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gui_harness as H


def scenario_form(args):
    """Build the real form and dump it. No filesystem, no subprocess."""
    gui = H.build_form()
    facts = {
        "controls": len(gui._all_args),
        "groups": list(gui.children),
        "required at top": [
            a.dest for a, w in gui._all_args.items() if a.required
        ],
    }
    return H.write_artifacts(args.out_dir, widget=gui.widget, gui=gui), facts


def scenario_runner(args):
    """Point a runner at an existing output directory and read everything back.

    This does not launch anything -- it reads the artifacts a completed run left
    behind, which is enough to exercise the history plot, epoch discovery and the
    viewer payloads. The replay stub (phase 1) drives the live path.
    """
    from abismal.gui.runner import AbismalRunner

    results = Path(args.results).expanduser()
    if not results.is_dir():
        sys.exit(f"--results must be a directory with abismal output; got {results}")

    runner = AbismalRunner(
        args=None, out_dir=str(results), has_phenix=True, total_epochs=args.epochs
    )
    runner._update_history()
    pdb_file, mtz_file = runner._find_latest_phenix_results()
    if pdb_file:
        runner._render_epoch(pdb_file, mtz_file)

    facts = {
        "out_dir": str(results),
        "history rows plotted": "yes" if runner.history_widget.outputs else "no (no history.csv?)",
        "latest epoch dir": Path(pdb_file).parent.name if pdb_file else None,
        "viewer initialized": runner._viewer_initialized,
        "progress": f"{runner.progress_widget.value}/{runner.progress_widget.max}",
    }
    extra = {}
    if pdb_file:
        from abismal.gui.components.gemmimol import GemmiMolViewer

        viewer = GemmiMolViewer(pdb_file=pdb_file, mtz_file=mtz_file)
        extra["viewer.html"] = viewer.html
        facts["map_keys"] = viewer.map_keys
    return H.write_artifacts(
        args.out_dir, widget=runner.to_widget(), runner=runner, extra=extra
    ), facts


def scenario_replay(args):
    """Launch the replay stub through the real start() path and snapshot the result.

    This is the live one: a real subprocess, real log tailing, real progress updates
    and a history plot that grows as it goes -- in a couple of seconds, with no GPU and
    no merge job.
    """
    out_dir = Path(args.out_dir)
    template = H.make_results_template(out_dir / "_template")
    runner = H.start_replay(
        out_dir / "run",
        results=template,
        delay=args.delay,
        total_epochs=args.epochs,
        has_phenix=True,
    )
    H.wait_for_replay(runner, timeout=args.timeout)

    facts = {
        "progress": f"{runner.progress_widget.value}/{runner.progress_widget.max}",
        "bar_style": runner.progress_widget.bar_style,
        "label": runner.progress_label.value,
        "stop button disabled": runner.stop_button.disabled,
        "log lines": len(H.log_text(runner).splitlines()),
        "pid file cleaned up": not Path(runner.pid_file).exists(),
    }
    pdb_file, mtz_file = runner._find_latest_phenix_results()
    facts["latest epoch dir"] = Path(pdb_file).parent.name if pdb_file else None

    extra = {}
    if pdb_file:
        from abismal.gui.components.gemmimol import GemmiMolViewer

        viewer = GemmiMolViewer(pdb_file=pdb_file, mtz_file=mtz_file)
        extra["viewer.html"] = viewer.html
        facts["map_keys"] = viewer.map_keys

    written = H.write_artifacts(
        args.out_dir, widget=runner.to_widget(), runner=runner, extra=extra
    )
    H.quiesce(runner)
    return written, facts


def scenario_viewer(args):
    """Just the standalone 3D viewer document, for the browser tier to load."""
    from abismal.gui.components.gemmimol import GemmiMolViewer

    viewer = GemmiMolViewer(pdb_file=args.pdb, mtz_file=args.mtz)
    facts = {"pdb": args.pdb, "mtz": args.mtz, "map_keys": viewer.map_keys}
    return H.write_artifacts(
        args.out_dir, extra={"viewer.html": viewer.html}
    ), facts


SCENARIOS = {
    "form": scenario_form,
    "runner": scenario_runner,
    "replay": scenario_replay,
    "viewer": scenario_viewer,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="form")
    p.add_argument("--out-dir", default="/tmp/abismal-gui-snap")
    p.add_argument("--results", help="an abismal output directory (scenario=runner)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds between replayed log lines (scenario=replay)")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--pdb", help="scenario=viewer")
    p.add_argument("--mtz", help="scenario=viewer")
    args = p.parse_args(argv)

    started = time.time()
    written, facts = SCENARIOS[args.scenario](args)
    elapsed = time.time() - started

    lines = [
        f"# abismal GUI snapshot -- scenario `{args.scenario}`",
        "",
        f"built in {elapsed:.1f}s",
        "",
        "## facts",
        "",
    ]
    lines += [f"- **{k}**: {v}" for k, v in facts.items()]
    lines += ["", "## files", ""]
    lines += [
        f"- `{name}` ({path.stat().st_size} B)"
        for name, path in sorted(written.items())
    ]
    summary = "\n".join(lines) + "\n"
    (Path(args.out_dir) / "summary.md").write_text(summary)

    print(summary)
    print(f"artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
