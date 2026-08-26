"""The two figures the runner draws: training history and anomalous peak heights.

Matplotlib is exercised for real -- the runner renders into an Output widget's
`outputs` as a base64 PNG, so the test decodes what a browser would receive. The
Line2D objects are read back off the axes rather than the image, since colour and
dash pattern are the point and pixels are a poor way to assert on them.
"""
import base64
import io

import pandas as pd
import pytest

import gui_harness as H


HISTORY = """Epoch,loss,val_loss,CCpred,val_CCpred,wCCpred,val_wCCpred
1,35.0,27.8,0.37,0.64,0.72,0.73
2,21.6,18.4,0.83,0.87,0.94,0.95
3,16.1,14.2,0.95,0.95,0.96,0.96
"""

PEAKS_HEADER = (
    "chain,seqid,residue,name,dist,peak,peakz,score,scorez,"
    "cenx,ceny,cenz,coordx,coordy,coordz\n"
)


def write_peaks(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("peaks.csv").write_text(
        PEAKS_HEADER
        + "".join(
            f"{chain},{seqid},{residue},,0.5,0.15,{peakz},0.5,30.0,"
            f"1.0,2.0,3.0,1.0,2.0,3.0\n"
            for chain, seqid, residue, peakz in rows
        )
    )


def png_of(widget):
    """The PNG bytes a widget is currently showing, or None."""
    pngs = dict(H.extract_pngs(widget))
    if not pngs:
        return None
    return next(iter(pngs.values()))


# ---------------------------------------------------------------------------
# history: Dark2, and a training/validation pair sharing one colour
# ---------------------------------------------------------------------------

def axes_for(metrics, columns=None):
    """Draw one metric group onto a bare axis and hand back its lines."""
    from matplotlib.figure import Figure
    from abismal.gui.runner import _plot_metrics

    df = pd.read_csv(io.StringIO(HISTORY))
    if columns is not None:
        df = df[columns]
    ax = Figure().subplots()
    _plot_metrics(ax, df, metrics)
    return ax


def test_a_metric_and_its_validation_partner_share_a_colour():
    """The pairing is the whole point: two colours for four lines, not four."""
    ax = axes_for(["CCpred", "wCCpred"])
    by_label = {line.get_label(): line for line in ax.get_lines()}

    assert by_label["CCpred"].get_color() == by_label["val_CCpred"].get_color()
    assert by_label["wCCpred"].get_color() == by_label["val_wCCpred"].get_color()
    assert by_label["CCpred"].get_color() != by_label["wCCpred"].get_color()


def test_training_is_solid_and_validation_is_dashed():
    ax = axes_for(["CCpred", "wCCpred"])
    for line in ax.get_lines():
        expected = "--" if line.get_label().startswith("val_") else "-"
        assert line.get_linestyle() == expected, line.get_label()


def test_the_colours_come_from_dark2():
    import seaborn as sns

    dark2 = {tuple(round(c, 6) for c in rgb) for rgb in sns.color_palette("Dark2", 8)}
    ax = axes_for(["loss", "CCpred", "wCCpred"])
    for line in ax.get_lines():
        rgb = tuple(round(c, 6) for c in line.get_color()[:3])
        assert rgb in dark2, f"{line.get_label()} is not a Dark2 colour"


def test_a_metric_with_no_validation_partner_still_plots():
    ax = axes_for(["loss"], columns=["Epoch", "loss"])
    assert [line.get_label() for line in ax.get_lines()] == ["loss"]


def test_val_columns_are_not_treated_as_metrics_of_their_own():
    """Otherwise val_loss gets its own colour and the pairing falls apart."""
    from abismal.gui.runner import _base_metrics

    df = pd.read_csv(io.StringIO(HISTORY))
    assert _base_metrics(df, lambda c: "loss" in c.lower()) == ["loss"]
    assert _base_metrics(df, lambda c: "CC" in c) == ["CCpred", "wCCpred"]


def test_the_history_figure_reaches_the_widget(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "history.csv").write_text(HISTORY)

    runner = runner_factory(out_dir=str(out_dir))
    runner._update_history()

    png = png_of(runner.history_widget)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# peaks
# ---------------------------------------------------------------------------

def test_peaks_are_collected_from_both_refinement_backends(runner_factory, tmp_path):
    """phenix writes eff_*, torchref writes torchref_*, and both land in one plot."""
    out_dir = tmp_path / "run"
    write_peaks(out_dir / "eff_0_asu_0_epoch_1", [("A", 30, "CYS", 11.0)])
    write_peaks(out_dir / "torchref_0_asu_0_epoch_2", [("A", 30, "CYS", 14.0)])

    peaks = runner_factory(out_dir=str(out_dir))._read_peaks()

    assert sorted(peaks["Epoch"]) == [1, 2]
    assert set(peaks["Residue"]) == {"CYS-30:A"}


def test_the_epoch_comes_from_the_directory_name(runner_factory, tmp_path):
    """peaks.csv carries no epoch column; the directory is the only record of it."""
    out_dir = tmp_path / "run"
    write_peaks(out_dir / "torchref_0_asu_0_epoch_7", [("A", 80, "CYS", 9.0)])

    peaks = runner_factory(out_dir=str(out_dir))._read_peaks()

    assert list(peaks["Epoch"]) == [7]


def test_no_peaks_is_not_an_error(runner_factory, tmp_path):
    """The overwhelmingly common case: a non-anomalous run, or one with no
    refinement at all. Nothing is drawn and nothing raises."""
    runner = runner_factory(out_dir=str(tmp_path), has_phenix=True)

    assert runner._read_peaks() is None
    runner._update_peaks()
    assert png_of(runner.peaks_widget) is None
    assert runner.peaks_label.layout.display == "none"


def test_a_malformed_peaks_file_is_skipped_not_fatal(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    good = out_dir / "torchref_0_asu_0_epoch_2"
    write_peaks(good, [("A", 30, "CYS", 11.0)])
    bad = out_dir / "torchref_0_asu_0_epoch_1"
    bad.mkdir(parents=True)
    (bad / "peaks.csv").write_text("something,else\n1,2\n")

    peaks = runner_factory(out_dir=str(out_dir))._read_peaks()

    assert list(peaks["Epoch"]) == [2]


def test_peaks_seen_in_too_few_epochs_are_dropped(runner_factory, tmp_path):
    """A one-off excursion is noise, not a site. The floor scales with how many
    epochs have been seen, so it filters during a run rather than only after."""
    out_dir = tmp_path / "run"
    for epoch in range(1, 11):
        rows = [("A", 30, "CYS", 10.0 + epoch)]
        if epoch == 4:
            rows.append(("A", 99, "GLY", 6.0))
        write_peaks(out_dir / f"torchref_0_asu_0_epoch_{epoch}", rows)

    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    runner._update_peaks()

    assert "GLY-99:A" in set(runner._read_peaks()["Residue"])   # it was read
    plotted = _legend_labels(runner)
    assert "CYS-30:A" in plotted
    assert "GLY-99:A" not in plotted                            # but not plotted


def _legend_labels(runner):
    """Re-draw the peak plot and read the residue names off the legend.

    The widget holds a PNG, so the labels are recovered by rebuilding the figure
    the same way _update_peaks does rather than by reading pixels.
    """
    import seaborn as sns
    from matplotlib.figure import Figure

    data = runner._read_peaks()
    n_epochs = data["Epoch"].nunique()
    min_points = max(1, round(0.5 * n_epochs))
    counts = data.groupby("Residue")["Epoch"].transform("size")
    data = data[counts >= min_points]

    ax = Figure().subplots()
    sns.lineplot(data, x="Epoch", y="peakz", hue="Residue", palette="Dark2", ax=ax)
    return {text.get_text() for text in ax.get_legend().get_texts()}


def test_drawing_peaks_reveals_the_section(runner_factory, tmp_path):
    """The header is hidden until there is something under it, because only an
    anomalous refinement run produces peaks and that cannot be known up front."""
    out_dir = tmp_path / "run"
    for epoch in (1, 2, 3):
        write_peaks(out_dir / f"eff_0_asu_0_epoch_{epoch}",
                    [("A", 30, "CYS", 10.0 + epoch)])

    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    assert runner.peaks_label.layout.display == "none"

    runner._update_peaks()

    assert runner.peaks_label.layout.display == ""
    assert png_of(runner.peaks_widget) is not None


def test_peaks_are_not_polled_for_a_run_without_refinement(runner_factory, tmp_path):
    """_poll only reaches _update_peaks through the phenix branch; there is no
    peaks.csv to find otherwise, and globbing for one every poll is wasted."""
    out_dir = tmp_path / "run"
    write_peaks(out_dir / "eff_0_asu_0_epoch_1", [("A", 30, "CYS", 11.0)])

    runner = runner_factory(out_dir=str(out_dir), has_phenix=False)
    runner._poll()

    assert png_of(runner.peaks_widget) is None


def test_the_peak_plot_is_not_redrawn_when_nothing_changed(runner_factory, tmp_path):
    """The poll runs on a timer. Rendering a seaborn figure every tick made the
    replay suite six times slower, and a finished run would redraw the same
    figure forever."""
    out_dir = tmp_path / "run"
    for epoch in (1, 2, 3):
        write_peaks(out_dir / f"eff_0_asu_0_epoch_{epoch}",
                    [("A", 30, "CYS", 10.0 + epoch)])

    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    runner._update_peaks()
    first = runner.peaks_widget.outputs

    runner._update_peaks()
    assert runner.peaks_widget.outputs is first   # not re-rendered

    write_peaks(out_dir / "eff_0_asu_0_epoch_4", [("A", 30, "CYS", 14.0)])
    runner._update_peaks()
    assert runner.peaks_widget.outputs is not first


def test_two_threads_can_draw_at_once(runner_factory, tmp_path):
    """The tailer calls _poll directly while _schedule_poll's Timer calls it on its
    own thread, so both plots really are drawn concurrently. Matplotlib's unsafe
    state is global -- the mathtext parser's pyparsing cache above all -- and the
    peak plot's "$\\sigma$" label goes straight through it, so without a lock this
    raises out of a daemon thread where nothing is watching.
    """
    import threading

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "history.csv").write_text(HISTORY)
    for epoch in range(1, 5):
        write_peaks(out_dir / f"eff_0_asu_0_epoch_{epoch}",
                    [("A", 30, "CYS", 10.0 + epoch), ("A", 80, "CYS", 8.0 + epoch)])

    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)

    failures = []

    def draw():
        for _ in range(6):
            try:
                # Clear the guard each time, or only the first pass renders and the
                # threads never overlap where it matters.
                runner._peaks_signature = None
                runner._update_peaks()
                runner._update_history()
            except Exception as error:
                failures.append(error)
                return

    threads = [threading.Thread(target=draw) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert not any(t.is_alive() for t in threads)
    assert not failures, failures
    assert png_of(runner.peaks_widget) is not None
