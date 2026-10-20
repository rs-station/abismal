"""The two figures the runner draws: training history and anomalous peak heights.

Matplotlib is exercised for real -- the runner renders into an Output widget's
`outputs` as a base64 PNG, so the test decodes what a browser would receive. The
Line2D objects are read back off the axes rather than the image, since colour and
dash pattern are the point and pixels are a poor way to assert on them.
"""
import base64
import io
from pathlib import Path

import matplotlib.colors
import numpy as np
import pandas as pd
import pytest

import gui_harness as H


# The shape abismal actually writes, down to the all-NaN row its pre-training
# callback emits before epoch 1.
HISTORY = """Epoch,loss,val_loss,NLL,val_NLL,KL,val_KL,KL_Σ,val_KL_Σ,CCpred,val_CCpred,wCCpred,val_wCCpred
0,,,,,,,,,,,,
1,35.0,27.8,30.9,22.9,0.61,1.36,3.10,3.31,0.37,0.64,0.72,0.73
2,21.6,18.4,15.4,11.8,2.55,3.02,3.62,3.71,0.83,0.87,0.94,0.95
3,16.1,14.2,9.10,8.30,3.31,3.30,3.85,3.88,0.95,0.95,0.96,0.96
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


# ---------------------------------------------------------------------------
# the loss panel: every term of the objective, on a log axis
# ---------------------------------------------------------------------------

def test_the_loss_panel_carries_the_terms_the_loss_is_made_of():
    """NLL and the KL terms are what `loss` sums to, and the only way to see
    which one a plateau or a divergence is coming from."""
    from abismal.gui.runner import _base_metrics, _is_loss_term

    df = pd.read_csv(io.StringIO(HISTORY))

    assert _base_metrics(df, _is_loss_term) == ["loss", "NLL", "KL", "KL_Σ"]


def test_the_cc_metrics_stay_out_of_the_loss_panel():
    from abismal.gui.runner import _is_loss_term

    for column in ("CCpred", "wCCpred", "Σ_mean", "Σ_std", "Epoch",
                   "Time (s)", "FB Used (MiB)"):
        assert not _is_loss_term(column), column


def test_the_loss_panel_is_log_scaled(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "history.csv").write_text(HISTORY)
    runner = runner_factory(out_dir=str(out_dir))

    loss_ax, cc_ax = _history_axes(runner)

    assert loss_ax.get_yscale() == "log"
    assert cc_ax.get_yscale() == "linear"   # CC runs through zero and near it


def test_a_negative_nll_does_not_cost_the_log_axis(runner_factory, tmp_path):
    """NLL goes negative routinely. Matplotlib clips those points and draws the
    rest, which is worth more than putting everything on a linear axis to keep
    them -- and it never raises, even with nothing positive left to draw."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    df = pd.read_csv(io.StringIO(HISTORY))
    df.loc[2, "NLL"] = -4.0
    df.loc[2, "val_NLL"] = -1.5
    df.to_csv(out_dir / "history.csv", index=False)

    runner = runner_factory(out_dir=str(out_dir))
    loss_ax, _ = _history_axes(runner)

    assert loss_ax.get_yscale() == "log"
    runner._update_history()
    assert png_of(runner.history_widget) is not None


# ---------------------------------------------------------------------------
# the legend
# ---------------------------------------------------------------------------

def test_the_legend_names_each_metric_once(runner_factory, tmp_path):
    """A val_ line has no colour of its own -- it is its metric's colour, dashed
    -- so listing both halves says the same thing twice, and doubled the legend
    when NLL and the KL terms joined the panel."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "history.csv").write_text(HISTORY)
    runner = runner_factory(out_dir=str(out_dir))

    loss_ax, _ = _history_axes(runner)
    labels = [text.get_text() for text in loss_ax.get_legend().get_texts()]

    assert labels == ["loss", "NLL", "KL", "KL_Σ", "training", "validation"]


def test_the_lines_keep_their_real_labels(runner_factory, tmp_path):
    """Only what the legend is built from is filtered; the artists are untouched,
    so the val_ lines stay identifiable."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "history.csv").write_text(HISTORY)
    runner = runner_factory(out_dir=str(out_dir))

    loss_ax, _ = _history_axes(runner)
    labels = {line.get_label() for line in loss_ax.get_lines()}

    assert {"loss", "val_loss", "KL_Σ", "val_KL_Σ"} <= labels


def _history_axes(runner):
    """The loss and CC axes, from the same function that draws them for real.

    _update_history hands its widget a PNG, so asserting on scales and legends
    means holding the figure itself -- and building an equivalent one here would
    be free to drift from the one that ships.
    """
    from abismal.gui.runner import _history_figure

    df = pd.read_csv(Path(runner.out_dir) / "history.csv")
    return _history_figure(df).get_axes()


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
    assert set(peaks["AtomType"]) == {"CYS"}


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


def _draw_peaks_axis(runner):
    """Re-draw the peak plot onto a bare axis.

    The widget holds a PNG, so figure properties are recovered by rebuilding
    the figure the same way _update_peaks does rather than by reading pixels.
    """
    from matplotlib.figure import Figure
    from abismal.gui.runner import _draw_peaks

    data = runner._read_peaks()
    n_epochs = data["Epoch"].nunique()
    min_points = max(1, round(0.5 * n_epochs))
    counts = data.groupby("Residue")["Epoch"].transform("size")
    data = data[counts >= min_points]

    ax = Figure().subplots()
    _draw_peaks(ax, data)
    return ax


def _legend_labels(runner):
    return {text.get_text() for text in _draw_peaks_axis(runner).get_legend().get_texts()}


def _line_with_ydata(ax, ydata):
    """The plotted (not legend-proxy) line whose y-values match ydata.

    Legend handles duplicate the real lines with their own Line2D objects, but
    those carry no data of their own, so matching on data picks out the real
    series regardless of how many proxies share the axis.
    """
    expected = tuple(float(v) for v in ydata)
    for line in ax.get_lines():
        if tuple(line.get_ydata()) == expected:
            return line
    raise AssertionError(f"no plotted line with ydata {ydata}")


def test_atom_type_sets_colour_and_residue_sets_linestyle(runner_factory, tmp_path):
    """Two CYS residues share a colour but get their own linestyle; MET gets its
    own colour. `name` (the atom name) is empty in every peaks.csv seen in
    practice, so `residue` -- CYS, MET, ... -- stands in as the atom type."""
    out_dir = tmp_path / "run"
    for epoch in (1, 2, 3):
        write_peaks(out_dir / f"eff_0_asu_0_epoch_{epoch}", [
            ("A", 30, "CYS", 10.0 + epoch),
            ("A", 80, "CYS", 20.0 + epoch),
            ("A", 12, "MET", 30.0 + epoch),
        ])

    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    ax = _draw_peaks_axis(runner)

    cys_30 = _line_with_ydata(ax, [11.0, 12.0, 13.0])
    cys_80 = _line_with_ydata(ax, [21.0, 22.0, 23.0])
    met_12 = _line_with_ydata(ax, [31.0, 32.0, 33.0])

    assert cys_30.get_color() == cys_80.get_color()
    assert cys_30.get_color() != met_12.get_color()
    assert cys_30.get_linestyle() != cys_80.get_linestyle()


def test_the_peak_axis_is_not_log_scaled(runner_factory, tmp_path):
    """A log scale used to compress away the early-epoch spread; peak heights
    are small and linear, so the axis should be too."""
    out_dir = tmp_path / "run"
    for epoch in (1, 2, 3):
        write_peaks(out_dir / f"eff_0_asu_0_epoch_{epoch}",
                    [("A", 30, "CYS", 10.0 + epoch)])

    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    ax = _draw_peaks_axis(runner)

    assert ax.get_yscale() == "linear"


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



# ---------------------------------------------------------------------------
# a lone point is visible
# ---------------------------------------------------------------------------

def test_a_one_epoch_metric_gets_a_marker_and_later_ones_do_not():
    """After the first epoch a line has one point and draws nothing; a marker
    shows it. From the second epoch on, the lines stand alone."""
    import pandas as pd
    from matplotlib.figure import Figure
    from abismal.gui.runner import _plot_metrics

    one = pd.DataFrame({"Epoch": [1], "loss": [3.0], "val_loss": [4.0]})
    ax = Figure().subplots()
    _plot_metrics(ax, one, ["loss"])
    assert all(line.get_marker() == "o" for line in ax.get_lines())

    two = pd.DataFrame({"Epoch": [1, 2], "loss": [3.0, 2.0], "val_loss": [4.0, 3.5]})
    ax = Figure().subplots()
    _plot_metrics(ax, two, ["loss"])
    assert all(line.get_marker() in (None, "None", "") for line in ax.get_lines())


def test_a_validation_series_with_one_value_gets_a_marker():
    """Validation can lag training (or appear once), so each series is judged on
    its own points."""
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    from abismal.gui.runner import _plot_metrics

    df = pd.DataFrame({"Epoch": [1, 2], "loss": [3.0, 2.0], "val_loss": [np.nan, 3.5]})
    ax = Figure().subplots()
    _plot_metrics(ax, df, ["loss"])
    markers = {line.get_label(): line.get_marker() for line in ax.get_lines()}
    assert markers["val_loss"] == "o"
    assert markers["loss"] in (None, "None", "")


def _markers(ax):
    """The scatter collections that mark lone points; seaborn's line plot adds
    error-band collections of its own, which are not markers."""
    from matplotlib.collections import PathCollection
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_first_epoch_peaks_are_drawn_as_markers(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    write_peaks(out_dir / "torchref_0_asu_0_epoch_1", [
        ("A", 30, "CYS", 11.0), ("A", 12, "MET", 31.0),
    ])
    ax = _draw_peaks_axis(runner_factory(out_dir=str(out_dir), has_phenix=True))

    points = np.concatenate([c.get_offsets() for c in _markers(ax)])
    assert sorted(points[:, 1].tolist()) == [11.0, 31.0]


def test_markers_share_the_colour_of_their_atom_type(runner_factory, tmp_path):
    """A site seen once so far sits beside sites with lines; its marker must be
    coloured like the lines of its atom type."""
    out_dir = tmp_path / "run"
    write_peaks(out_dir / "torchref_0_asu_0_epoch_1", [("A", 30, "CYS", 11.0)])
    write_peaks(out_dir / "torchref_0_asu_0_epoch_2", [
        ("A", 30, "CYS", 12.0), ("A", 80, "CYS", 21.0),
    ])
    ax = _draw_peaks_axis(runner_factory(out_dir=str(out_dir), has_phenix=True))

    cys_30 = _line_with_ydata(ax, [11.0, 12.0])
    assert len(_markers(ax)) == 1
    marker_colour = _markers(ax)[0].get_facecolors()[0][:3]
    assert np.allclose(marker_colour, matplotlib.colors.to_rgb(cys_30.get_color()))
    assert cys_30.get_marker() in (None, "None", "")


def test_sites_with_several_epochs_get_no_markers(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    for epoch in (1, 2, 3):
        write_peaks(out_dir / f"torchref_0_asu_0_epoch_{epoch}", [("A", 30, "CYS", 10.0 + epoch)])
    ax = _draw_peaks_axis(runner_factory(out_dir=str(out_dir), has_phenix=True))
    assert not _markers(ax)


# ---------------------------------------------------------------------------
# R-work / R-free beside the peak heights
# ---------------------------------------------------------------------------

def write_torchref_result(directory, rwork, rfree):
    """A torchref result directory as the worker leaves it: its R values close
    stdout.txt, and the PDB header carries the *input* model's (torchref before
    0.7.0 copied REMARK 3), which must not be what is plotted."""
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("stdout.txt").write_text(
        f"Cycle  5/5: Rwork={rwork + 0.01:.4f}  Rfree={rfree + 0.01:.4f}\n"
        f"Rwork={rwork:.4f}\nRfree={rfree:.4f}\nadp_aniso_sigma=0.2\n"
    )
    directory.joinpath("refined.pdb").write_text(
        "REMARK   3   R VALUE            (WORKING SET) : 0.9999\n"
        "REMARK   3   FREE R VALUE                     : 0.9999\n"
    )


def write_phenix_result(directory, rwork, rfree, log=True):
    directory.mkdir(parents=True, exist_ok=True)
    if log:
        directory.joinpath("refine_001.log").write_text(
            "Start R-work = 0.3000, R-free = 0.3100\n"
            f"Final R-work = {rwork:.4f}, R-free = {rfree:.4f}\n"
        )
    # The opening of a real phenix.refine REMARK 3 block: gemmi records a
    # refinement only once it has seen the DATA USED IN REFINEMENT section.
    directory.joinpath("refine_001.pdb").write_text(
        "REMARK   3\n"
        "REMARK   3 REFINEMENT.\n"
        "REMARK   3   PROGRAM     : PHENIX (1.21.2_5419: ???)\n"
        "REMARK   3\n"
        "REMARK   3  X-RAY DATA.\n"
        "REMARK   3  \n"
        "REMARK   3  REFINEMENT TARGET : ML\n"
        "REMARK   3  \n"
        "REMARK   3  DATA USED IN REFINEMENT.\n"
        "REMARK   3   RESOLUTION RANGE HIGH (ANGSTROMS) : 1.40    \n"
        "REMARK   3   RESOLUTION RANGE LOW  (ANGSTROMS) : 25.78   \n"
        "REMARK   3  \n"
        "REMARK   3  FIT TO DATA USED IN REFINEMENT.\n"
        f"REMARK   3   R VALUE            (WORKING SET) : {rwork:.4f}\n"
        f"REMARK   3   FREE R VALUE                     : {rfree:.4f}\n"
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1\n"
        "ATOM      1  CA  GLY A   1       1.000   1.000   1.000  1.00 20.00           C\n"
        "END\n"
    )


def test_torchref_r_values_come_from_the_worker_not_the_header(tmp_path):
    from abismal.gui.runner import _refinement_r_factors
    d = tmp_path / "torchref_0_asu_0_epoch_1"
    write_torchref_result(d, 0.1370, 0.1447)
    assert _refinement_r_factors(d) == (0.1370, 0.1447)


def test_phenix_r_values_come_from_the_log_or_else_the_model(tmp_path):
    from abismal.gui.runner import _refinement_r_factors
    with_log = tmp_path / "eff_0_asu_0_epoch_1"
    write_phenix_result(with_log, 0.2147, 0.2366)
    assert _refinement_r_factors(with_log) == (0.2147, 0.2366)
    without = tmp_path / "eff_0_asu_0_epoch_2"
    write_phenix_result(without, 0.2100, 0.2300, log=False)
    assert _refinement_r_factors(without) == pytest.approx((0.2100, 0.2300))


def test_an_unfinished_refinement_has_no_r_values(tmp_path):
    from abismal.gui.runner import _refinement_r_factors
    d = tmp_path / "torchref_0_asu_0_epoch_1"
    d.mkdir()
    d.joinpath("stdout.txt").write_text("Cycle  1/5: Rwork=0.2  Rfree=0.21\n")
    assert _refinement_r_factors(d) is None


def test_the_r_chart_appears_once_a_refinement_finishes(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    runner._update_r_factors()
    assert runner.r_factors_label.layout.display == "none"
    assert png_of(runner.r_factors_widget) is None

    write_torchref_result(out_dir / "torchref_0_asu_0_epoch_1", 0.20, 0.22)
    runner._update_r_factors()
    assert runner.r_factors_label.layout.display == ""
    assert png_of(runner.r_factors_widget) is not None


def test_a_non_anomalous_run_still_gets_r_values(runner_factory, tmp_path):
    """No peaks.csv, so no peak chart -- but the R chart applies to every run."""
    out_dir = tmp_path / "run"
    write_phenix_result(out_dir / "eff_0_asu_0_epoch_1", 0.21, 0.23)
    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    runner._poll()
    assert png_of(runner.r_factors_widget) is not None
    assert runner.peaks_label.layout.display == "none"


def test_the_r_chart_is_redrawn_only_when_a_result_changes(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    write_torchref_result(out_dir / "torchref_0_asu_0_epoch_1", 0.20, 0.22)
    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    runner._update_r_factors()
    first = runner.r_factors_widget.outputs
    runner._update_r_factors()
    assert runner.r_factors_widget.outputs is first
    write_torchref_result(out_dir / "torchref_0_asu_0_epoch_2", 0.18, 0.21)
    runner._update_r_factors()
    assert runner.r_factors_widget.outputs is not first


def _draw_r_axis(runner):
    from matplotlib.figure import Figure
    from abismal.gui.runner import _draw_r_factors
    ax = Figure().subplots()
    _draw_r_factors(ax, runner._read_r_factors())
    return ax


def test_rwork_is_solid_and_rfree_dashed_in_one_colour(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    for epoch, (w, f) in enumerate([(0.25, 0.27), (0.21, 0.24), (0.19, 0.23)], start=1):
        write_torchref_result(out_dir / f"torchref_0_asu_0_epoch_{epoch}", w, f)
    ax = _draw_r_axis(runner_factory(out_dir=str(out_dir), has_phenix=True))
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert list(lines["Rwork"].get_ydata()) == [0.25, 0.21, 0.19]
    assert list(lines["Rfree"].get_ydata()) == [0.27, 0.24, 0.23]
    assert lines["Rwork"].get_linestyle() == "-" and lines["Rfree"].get_linestyle() == "--"
    assert lines["Rwork"].get_color() == lines["Rfree"].get_color()
    assert all(line.get_marker() in (None, "None", "") for line in lines.values())


def test_one_epoch_of_r_values_is_marked(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    write_torchref_result(out_dir / "torchref_0_asu_0_epoch_1", 0.25, 0.27)
    ax = _draw_r_axis(runner_factory(out_dir=str(out_dir), has_phenix=True))
    assert all(line.get_marker() == "o" for line in ax.get_lines())


def test_each_refined_model_gets_its_own_colour(runner_factory, tmp_path):
    out_dir = tmp_path / "run"
    for epoch in (1, 2):
        write_torchref_result(out_dir / f"torchref_0_asu_0_epoch_{epoch}", 0.20, 0.22)
        write_torchref_result(out_dir / f"torchref_1_asu_0_epoch_{epoch}", 0.30, 0.32)
    ax = _draw_r_axis(runner_factory(out_dir=str(out_dir), has_phenix=True))
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert set(lines) == {"Rwork (torchref_0)", "Rfree (torchref_0)", "Rwork (torchref_1)", "Rfree (torchref_1)"}
    assert lines["Rwork (torchref_0)"].get_color() != lines["Rwork (torchref_1)"].get_color()


def test_the_post_training_watcher_keeps_both_charts_current(runner_factory, tmp_path, monkeypatch):
    """Refinement usually finishes after training: the last epoch's peaks and R
    values arrive while only the post-training watcher is still looking."""
    out_dir = tmp_path / "run"
    runner = runner_factory(out_dir=str(out_dir), has_phenix=True)
    runner.poll_interval = 0
    runner._monitoring_active = True
    write_torchref_result(out_dir / "torchref_0_asu_0_epoch_1", 0.20, 0.22)
    write_peaks(out_dir / "torchref_0_asu_0_epoch_1", [("A", 30, "CYS", 11.0)])
    runner._post_training_phenix_watcher(max_unchanged=1)
    assert png_of(runner.r_factors_widget) is not None
    assert png_of(runner.peaks_widget) is not None


def test_a_model_header_without_r_values_gives_none(tmp_path):
    """gemmi reports a missing R value as NaN; that must not reach the plot."""
    from abismal.gui.runner import _refinement_r_factors
    d = tmp_path / "eff_0_asu_0_epoch_1"
    write_phenix_result(d, 0.21, 0.23, log=False)
    pdb = d / "refine_001.pdb"
    pdb.write_text("".join(l for l in pdb.read_text().splitlines(True) if "R VALUE" not in l))
    assert _refinement_r_factors(d) is None
