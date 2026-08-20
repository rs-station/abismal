"""Periodic peak_local_max: pad with periodic images, find, crop back."""
import numpy as np
from skimage.feature import peak_local_max

def periodic_peaks(arr, min_distance, threshold_abs, pad=None):
    """Local maxima of a periodic map, in original-array index space.

    Pad with wrapped copies so every voxel inside the real cell has its true
    periodic neighbourhood, run peak_local_max, then CROP to the real extent.
    Cropping (not folding) is what makes this exact: a padding-region detection
    is by construction a duplicate of one inside the cell, so discarding it
    loses nothing and cannot double-count.
    """
    if pad is None:
        pad = int(min_distance)
    ap = np.pad(arr, pad, mode="wrap")
    pk = peak_local_max(ap, min_distance=min_distance,
                        threshold_abs=threshold_abs, exclude_border=False)
    n = np.array(arr.shape)
    inside = np.all((pk >= pad) & (pk < pad + n), axis=1)
    return pk[inside] - pad


def dedup_symmetry(idx, shape, spacegroup):
    """Keep one representative per symmetry-equivalent set of peaks.

    peak_local_max sees a P1 map and reports every symmetry copy; gemmi's flood
    fill dedupes internally with an ASU mask. Canonicalise instead of masking:
    map each peak through all operators, take the lexicographically smallest
    grid index as the key. A peak whose maximum straddles the ASU boundary would
    be dropped outright by masking; canonicalising is boundary-safe.

    `idx` must already be sorted by descending peak height, so the survivor of
    each group is its strongest member.
    """
    if len(idx) == 0:
        return idx
    ops = list(spacegroup.operations())
    n = np.array(shape, dtype=float)
    ni = np.array(shape)
    seen, keep = set(), []
    for i, f in enumerate(idx / n):
        key = min(
            tuple(np.mod(np.round(np.mod(op.apply_to_xyz(list(f)), 1.0) * n).astype(int), ni))
            for op in ops
        )
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return idx[keep]


# --------------------------------------------------------------------------
# Prototype only -- not wired into the worker. Requires scikit-image, which is
# NOT in the abismal-torchref env (it was benchmarked under opt/conda_base).
#
# Measured 2026-08-20 against rsbooster's gemmi flood fill, both on
# cxidb_81_small and hewl epoch-100 anomalous maps at sample_rate 5:
#
#   correctness  exactly roll-invariant (9 shifts) with the crop approach;
#                results identical from threshold 0 sigma to 5 sigma, so the
#                detect cutoff can equal the reporting cutoff -- no margin
#                heuristic, and gemmi's 3-voxel blob floor cannot apply.
#   resolving    hewl: 10 peaks vs flood fill's 6. Flood fill merges each
#                disulfide's two sulfurs (2.02-2.05 A apart) into one blob;
#                this resolves both. cxidb_81_small: identical 6 peaks and
#                identical z-scores.
#   speed        0.54 s (cxidb, 22.1M voxels) / 0.17 s (hewl) at 5 sigma, vs
#                ~0.10 s for flood fill. Linear in voxel count. Rises steeply
#                below 3 sigma (7.4 s at 0 sigma) but there is no reason to go
#                there.
#   memory       ~236 MB python-side peak for an 88 MB map (wrap pad + skimage's
#                maximum filter, both full-size copies).
#
# Two things the naive approach gets wrong and this module fixes:
#   1. CROP, do not fold. Mapping padding-region detections back with a modulo
#      adds spurious peaks; a padding detection is by construction a duplicate
#      of one inside the cell, so discard it.
#   2. Symmetry. peak_local_max sees a P1 map and returns all 12 copies of each
#      peak; gemmi dedupes internally with an ASU mask. dedup_symmetry does it
#      by canonicalisation.
#
# Open choices before adopting:
#   - min_distance=3 voxels works on both maps. >=6 merges HOH1002 into ZN317
#     (3.31 A apart), so it must stay below the closest pair worth resolving.
#   - Position is the maximum voxel, not the intensity-weighted centroid the
#     flood fill reports (ZN317: 0.290 A vs 0.157 A from the atom). The centroid
#     is the better position estimate; the maximum is what the z-score means.
#     Consider reporting a centroid over the peak's neighbourhood.
# --------------------------------------------------------------------------
