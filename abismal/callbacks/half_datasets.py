"""
CC1/2 and CCanom between random half datasets, computed during training.

The statistics come from `VariationalMergingModel.half_posterior`: two copies of
the surrogate posterior, each fit to the images of one random half of the data
with the full model's scales. Correlations between their structure factor
estimates measure how reproducible the merged data are, which needs no model
of the structure -- and, unlike the training or validation likelihood, cannot
be raised by the model fitting noise, which does not reproduce across halves.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from reciprocalspaceship.utils import bin_by_percentile

from abismal.callbacks.base import Callback


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


def _spearman(x, y):
    return spearmanr(x, y).statistic if len(x) > 2 else np.nan


def _correlations(prefix, a, b, w=None):
    if len(a) < 3:
        return {f"{prefix}": np.nan, f"{prefix}_spearman": np.nan}
    out = {
        f"{prefix}": weighted_pearson(a, b, w),
        f"{prefix}_spearman": _spearman(a, b),
    }
    if w is not None:
        out[f"{prefix}_unweighted"] = weighted_pearson(a, b)
    return out


def _numpy(x):
    return x.detach().cpu().numpy()


def half_dataset_pairs(model):
    """
    Paired half-dataset estimates.

    Returns
    -------
    intensities : pandas.DataFrame
        One row per reflection seen in both halves: dHKL, I and SIGI of each half.
    anomalous : pandas.DataFrame
        One row per acentric Friedel pair whose two members were both seen in
        both halves: dHKL and the anomalous difference dI = I(+) - I(-) of each
        half with its standard deviation.
    """
    hp = model.half_posterior
    rac = model.surrogate_posterior.rac
    n_asu = len(rac)
    I, SIGI = map(_numpy, hp.get_flat_isigi())
    seen = _numpy(hp.seen)
    asu_ids = _numpy(hp.rac.asu_id)

    intensities, anomalous = [], []
    for i, rasu in enumerate(rac):
        a = np.flatnonzero(asu_ids == i)
        b = np.flatnonzero(asu_ids == i + n_asu)
        both = seen[a] & seen[b]
        intensities.append(pd.DataFrame({
            "dHKL": rasu.dHKL[both],
            "I_A": I[a][both], "SIGI_A": SIGI[a][both],
            "I_B": I[b][both], "SIGI_B": SIGI[b][both],
        }))
        if not rasu.anomalous:
            continue
        # In an anomalous ASU an acentric reflection and its Friedel mate are
        # separate entries; centric reflections map onto themselves.
        mate = rasu.miller_ids(-rasu.Hunique)
        j = np.arange(rasu.asu_size)
        pair = (mate > j)  # each acentric pair once
        j, m = j[pair], mate[pair]
        ok = both[j] & both[m]
        j, m = j[ok], m[ok]
        anomalous.append(pd.DataFrame({
            "dHKL": rasu.dHKL[j],
            "dI_A": I[a][j] - I[a][m], "SIGdI_A": np.hypot(SIGI[a][j], SIGI[a][m]),
            "dI_B": I[b][j] - I[b][m], "SIGdI_B": np.hypot(SIGI[b][j], SIGI[b][m]),
        }))
    intensities = pd.concat(intensities, ignore_index=True)
    anomalous = pd.concat(anomalous, ignore_index=True) if anomalous else None
    return intensities, anomalous


def _statistics(intensities, anomalous):
    out = {"N½": len(intensities)}
    out.update(_correlations("CC½", intensities.I_A.to_numpy(), intensities.I_B.to_numpy()))
    if anomalous is not None:
        out["Nanom"] = len(anomalous)
        w = anomalous_weights(anomalous.SIGdI_A.to_numpy(), anomalous.SIGdI_B.to_numpy())
        out.update(_correlations("CCanom", anomalous.dI_A.to_numpy(), anomalous.dI_B.to_numpy(), w))
    return out


def half_dataset_statistics(model, bins=10):
    """
    CC1/2 (Pearson and Spearman) and, for anomalous data, CCanom (weighted
    and unweighted Pearson, and Spearman), overall and in `bins` resolution
    shells of equal numbers of reflections.

    Returns
    -------
    overall : dict
    binned : pandas.DataFrame
        One row per resolution shell, with its dHKL limits.
    """
    intensities, anomalous = half_dataset_pairs(model)
    overall = _statistics(intensities, anomalous)

    rows = []
    if len(intensities) >= bins:
        labels, edges = bin_by_percentile(intensities.dHKL.to_numpy(), bins=bins, ascending=False)
        intensities = intensities.assign(bin=labels)
        if anomalous is not None:
            # The same shells, so both statistics line up by row
            anomalous = anomalous.assign(bin=_bin_like(anomalous.dHKL.to_numpy(), edges))
        for k in range(bins):
            sel_i = intensities[intensities.bin == k]
            sel_a = None if anomalous is None else anomalous[anomalous.bin == k]
            row = {"bin": k, "d_max": max(edges[k], edges[k + 1]), "d_min": min(edges[k], edges[k + 1])}
            row.update(_statistics(sel_i, sel_a))
            rows.append(row)
    return overall, pd.DataFrame(rows)


def _bin_like(d, edges):
    """Assign resolutions `d` to the shells bounded by `edges` (either order)."""
    lo = np.minimum(edges[:-1], edges[1:])
    hi = np.maximum(edges[:-1], edges[1:])
    labels = np.full(len(d), -1)
    for k, (l, h) in enumerate(zip(lo, hi)):
        labels[(d >= l) & (d <= h) & (labels < 0)] = k
    return labels


class HalfDatasetStatistics(Callback):
    """
    Record CC1/2 and CCanom between the half posteriors after every epoch.

    The overall statistics are added to the epoch's logs, so they reach
    history.csv and can drive EarlyStopping (place this callback before it).
    The per-shell statistics are appended to
    `output_directory/half_dataset_statistics.csv`.
    """
    def __init__(self, output_directory, bins=10):
        self.output_directory = output_directory
        self.bins = bins

    def on_epoch_end(self, epoch, logs=None):
        if getattr(self.model, "half_posterior", None) is None:
            return
        overall, binned = half_dataset_statistics(self.model, self.bins)
        for k, v in overall.items():
            if logs is not None:
                logs[k] = v
            self.trainer.history.setdefault(k, []).append(v)
        if len(binned):
            binned.insert(0, "Epoch", epoch + 1)
            out = os.path.join(self.output_directory, "half_dataset_statistics.csv")
            binned.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
