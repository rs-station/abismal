"""
CC1/2 and CCanom between random half datasets, computed during training.

The half datasets come from `VariationalMergingModel.half_posterior`: two copies
of the surrogate posterior, each fit to the images of one random half of the
data with the full model's scales, as `abismal.cchalf` fits them afterwards with
the final scales. Every epoch the two half merges are written out in cchalf's
layout and correlated.

The statistics are computed from merged datasets, so the same functions apply
to the half merges written here and to the output of `abismal.cchalf`.

Note that the halves share the full model's scales, so errors the scale model
makes the same way in both -- including those of a model that overfits -- read
as reproducible signal. As with conventional CC1/2 after joint scaling, these
measure the precision of the merge, not its accuracy; on HEWL and cxidb_81 they
kept rising while the anomalous signal of an overfitting model fell.
"""
import os

import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from scipy.stats import spearmanr
from reciprocalspaceship.utils import bin_by_percentile

from abismal.callbacks.base import Callback

ANOMALOUS_COLUMNS = ["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"]


def weighted_pearson(x, y, w=None):
    """Pearson correlation of `x` and `y` with optional weights `w`."""
    if w is None:
        w = np.ones_like(x)
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    dx, dy = x - mx, y - my
    cov = (w * dx * dy).sum()
    return cov / np.sqrt((w * dx * dx).sum() * (w * dy * dy).sum())


def anomalous_weights(sig_a, sig_b):
    """
    Weights for CCanom: the inverse variance of the pair of anomalous
    differences being correlated, 1 / (sigma_A^2 + sigma_B^2).
    """
    return 1. / (np.square(sig_a) + np.square(sig_b))


def _stacked_intensities(ds):
    """I and SIGI with each Friedel mate of an anomalous dataset as its own row."""
    if all(k in ds for k in ANOMALOUS_COLUMNS):
        ds = ds[ANOMALOUS_COLUMNS].stack_anomalous()
    return ds[["I", "SIGI"]].dropna()


def pair_datasets(ds_a, ds_b):
    """
    Pair two half-dataset merges reflection by reflection.

    Parameters
    ----------
    ds_a, ds_b : reciprocalspaceship.DataSet
        Merged half datasets indexed by miller index, with I and SIGI columns,
        or I(+), SIGI(+), I(-), SIGI(-) for anomalous data -- the layout abismal
        writes.

    Returns
    -------
    intensities : pandas.DataFrame
        dHKL and I, SIGI of each half, for every reflection (each Friedel mate
        separately) present in both.
    anomalous : pandas.DataFrame or None
        dHKL and the anomalous difference dI = I(+) - I(-) with its standard
        deviation in each half, for acentric reflections with both mates present
        in both halves. None for data without anomalous columns.
    """
    a, b = _stacked_intensities(ds_a), _stacked_intensities(ds_b)
    joined = a.join(b, how="inner", lsuffix="_A", rsuffix="_B")
    joined.cell, joined.spacegroup = ds_a.cell, ds_a.spacegroup
    intensities = pd.DataFrame({
        "dHKL": joined.compute_dHKL()["dHKL"].to_numpy(),
        "I_A": joined["I_A"].to_numpy(float), "SIGI_A": joined["SIGI_A"].to_numpy(float),
        "I_B": joined["I_B"].to_numpy(float), "SIGI_B": joined["SIGI_B"].to_numpy(float),
    })

    anomalous = None
    if all(k in ds_a for k in ANOMALOUS_COLUMNS) and all(k in ds_b for k in ANOMALOUS_COLUMNS):
        def differences(ds):
            ds = ds[ANOMALOUS_COLUMNS].dropna()
            ds = ds[~ds.label_centrics()["CENTRIC"]]
            return pd.DataFrame({
                "dI": (ds["I(+)"] - ds["I(-)"]).to_numpy(float),
                "SIGdI": np.hypot(ds["SIGI(+)"].to_numpy(float), ds["SIGI(-)"].to_numpy(float)),
            }, index=ds.index)
        pairs = differences(ds_a).join(differences(ds_b), how="inner", lsuffix="_A", rsuffix="_B")
        dhkl = rs.DataSet(pairs, cell=ds_a.cell, spacegroup=ds_a.spacegroup).compute_dHKL()["dHKL"]
        anomalous = pd.DataFrame({
            "dHKL": dhkl.to_numpy(),
            "dI_A": pairs["dI_A"].to_numpy(), "SIGdI_A": pairs["SIGdI_A"].to_numpy(),
            "dI_B": pairs["dI_B"].to_numpy(), "SIGdI_B": pairs["SIGdI_B"].to_numpy(),
        })
    return intensities, anomalous


def _spearman(x, y):
    return spearmanr(x, y).statistic if len(x) > 2 else np.nan


def _correlations(prefix, a, b, w=None):
    if len(a) < 3:
        out = {f"{prefix}": np.nan, f"{prefix}_spearman": np.nan}
        if w is not None:
            out[f"{prefix}_unweighted"] = np.nan
        return out
    out = {
        f"{prefix}": weighted_pearson(a, b, w),
        f"{prefix}_spearman": _spearman(a, b),
    }
    if w is not None:
        out[f"{prefix}_unweighted"] = weighted_pearson(a, b)
    return out


def _statistics(intensities, anomalous):
    out = {"N½": len(intensities)}
    out.update(_correlations("CC½", intensities.I_A.to_numpy(), intensities.I_B.to_numpy()))
    if anomalous is not None:
        out["Nanom"] = len(anomalous)
        w = anomalous_weights(anomalous.SIGdI_A.to_numpy(), anomalous.SIGdI_B.to_numpy())
        out.update(_correlations("CCanom", anomalous.dI_A.to_numpy(), anomalous.dI_B.to_numpy(), w))
    return out


def _bin_like(d, edges):
    """Assign resolutions `d` to the shells bounded by `edges` (either order)."""
    lo = np.minimum(edges[:-1], edges[1:])
    hi = np.maximum(edges[:-1], edges[1:])
    labels = np.full(len(d), -1)
    for k, (l, h) in enumerate(zip(lo, hi)):
        labels[(d >= l) & (d <= h) & (labels < 0)] = k
    return labels


def half_dataset_statistics(intensities, anomalous=None, bins=10):
    """
    CC1/2 (Pearson and Spearman) and, for anomalous data, CCanom (weighted
    and unweighted Pearson, and Spearman), overall and in `bins` resolution
    shells of equal numbers of reflections.

    Parameters
    ----------
    intensities, anomalous : pandas.DataFrame
        As returned by `pair_datasets`; concatenate them to pool several ASUs.

    Returns
    -------
    overall : dict
    binned : pandas.DataFrame
        One row per resolution shell, with its dHKL limits.
    """
    overall = _statistics(intensities, anomalous)
    rows = []
    if len(intensities) >= bins:
        labels, edges = bin_by_percentile(intensities.dHKL.to_numpy(), bins=bins, ascending=False)
        intensities = intensities.assign(bin=labels)
        if anomalous is not None:
            # The same shells, so both statistics line up by row
            anomalous = anomalous.assign(bin=_bin_like(anomalous.dHKL.to_numpy(), edges))
        for k in range(bins):
            sel_a = None if anomalous is None else anomalous[anomalous.bin == k]
            row = {"bin": k, "d_max": max(edges[k], edges[k + 1]), "d_min": min(edges[k], edges[k + 1])}
            row.update(_statistics(intensities[intensities.bin == k], sel_a))
            rows.append(row)
    return overall, pd.DataFrame(rows)


def half_datasets(model, dmin=None):
    """
    The two half merges of every ASU of `model`, as `abismal.cchalf` would
    write them: the reflections each half has observed, truncated to `dmin`.

    Returns
    -------
    list of (reciprocalspaceship.DataSet, reciprocalspaceship.DataSet)
        One (half 1, half 2) pair per ASU.
    """
    n = len(model.surrogate_posterior.rac)
    datasets = list(model.half_posterior.to_datasets(seen=True))
    if dmin is not None:
        datasets = [ds[ds.compute_dHKL()["dHKL"] >= dmin] for ds in datasets]
    return [(datasets[i], datasets[i + n]) for i in range(n)]


class HalfDatasetStatistics(Callback):
    """
    After every epoch, write the half-dataset merges and record CC1/2 and CCanom
    between them.

    The merges go to `output_directory/half_1/asu_{i}_epoch_{n}.mtz` and
    `half_2/...`, the layout `abismal.cchalf` uses. The overall statistics are
    added to the epoch's logs, so they reach history.csv and EarlyStopping
    (place this callback before it); the per-shell statistics are appended to
    `output_directory/half_dataset_statistics.csv`.
    """
    def __init__(self, output_directory, dmin=None, bins=10):
        self.output_directory = output_directory
        self.dmin = dmin
        self.bins = bins

    def on_epoch_end(self, epoch, logs=None):
        if getattr(self.model, "half_posterior", None) is None:
            return
        intensities, anomalous = [], []
        for asu, halves in enumerate(half_datasets(self.model, self.dmin)):
            for k, ds in enumerate(halves):
                out_dir = os.path.join(self.output_directory, f"half_{k + 1}")
                os.makedirs(out_dir, exist_ok=True)
                ds.write_mtz(os.path.join(out_dir, f"asu_{asu}_epoch_{epoch + 1}.mtz"))
            i, a = pair_datasets(*halves)
            intensities.append(i)
            if a is not None:
                anomalous.append(a)
        intensities = pd.concat(intensities, ignore_index=True)
        anomalous = pd.concat(anomalous, ignore_index=True) if anomalous else None
        overall, binned = half_dataset_statistics(intensities, anomalous, self.bins)
        for k, v in overall.items():
            if logs is not None:
                logs[k] = v
            self.trainer.history.setdefault(k, []).append(v)
        if len(binned):
            binned.insert(0, "Epoch", epoch + 1)
            out = os.path.join(self.output_directory, "half_dataset_statistics.csv")
            binned.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
