"""Standalone torchref refinement worker.

Invoked by :class:`abismal.callbacks.torchref.TorchRefRunner` as a detached
subprocess *by file path* (``python /path/to/_torchref_worker.py ...``). Running
it by path — rather than ``python -m abismal.callbacks._torchref_worker`` —
deliberately avoids importing the ``abismal`` package, whose ``__init__`` pulls
in TensorFlow/tf_keras. This keeps the worker a pure PyTorch process so it does
not initialize a second TF/CUDA context alongside the training process.

Basic refinement workflow: reset every atomic B-factor to a flat value, then run
ADP (B-factor) refinement against the merged structure factors. Anomalous vs
non-anomalous is auto-detected from the MTZ columns. Heavier refinement
protocols are intended to be supplied later via a user-provided Python script.

Anomalous data is passed through as F(+)/F(-) so torchref refines against both
Friedel mates. It then writes ANOM/PANOM difference map coefficients itself, and
the worker runs peak finding on those against the refined model -- the torchref
equivalent of what :class:`abismal.callbacks.peak_finder.AnomalousPeakFinder`
does with phenix.

Requires torchref >= 0.6.x. Up to 0.5.2, loading canonicalized HKLs with
``include_friedel=True`` and kept no signed index, so Friedel mates folded onto
one reflection and no anomalous refinement was possible; this worker used to
average the mates to work around that. 0.6.x keeps a signed ``hkl_anomalous``
alongside the canonical ``hkl``, which made the workaround obsolete -- and
costly: on hewl, averaging gave Rwork/Rfree 0.1671/0.1627 against 0.1589/0.1583
for the two mates refined as independent observations.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Invoked by file path, Python prepends this script's directory
# (abismal/callbacks/) to sys.path, where the sibling ``torchref.py`` callback
# module would shadow the installed ``torchref`` package. Drop that entry so
# ``from torchref import ...`` resolves to the real package.
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _here)]

# Flat starting B-factor (A^2) every atom is reset to before ADP refinement, so
# the reference model's already-refined B-factors cannot flatter the R-factors.
#
# A flat reset used to be actively harmful: torchref <=0.5.2 regularised ADPs
# with a log-normal KL (`Model.adp_kl_divergence_loss`) containing
# -log(std(log B)), so giving every atom the same B drove std(log B) to zero and
# the term to +inf. On hewl that meant KL = 12.45 against 1.05 for the PDB's own
# B-factors, ~125 "non-finite loss" warnings per run, and ADP refinement that
# could not move. torchref 0.6.x replaced that restraint with an inverse-gamma
# NLL (`adp_sigd_math`) which is finite for a perfectly uniform B distribution,
# so the flat reset is fine again -- verified 2026-08-14 on 0.6.3: 2 warnings
# per run and Rwork 0.2624 -> 0.1674 in two macrocycles. Keep torchref >=0.6.x.
RESET_B = 20.0

# Number of scaling + ADP-refinement macrocycles.
MACRO_CYCLES = 5

# Candidate values for SIMU's deviatoric sigma when --adp-aniso-sigma is "auto".
#
# Going isotropic -> anisotropic adds 5,560 parameters and *every one of them is
# deviatoric*; the 1,210 magnitude parameters are unchanged. So the dial that
# matters is the one on tensor shape, not the `adp` group weight -- that weight
# scales the magnitude and anisotropy channels together, and turning it up to
# reach the under-restrained tensors also clamps B magnitudes that were already
# fine. Measured on hewl 2026-08-14, targeting anisotropy directly is better on
# both axes at once:
#
#     group weight 0.05 :  Rfree 0.1468, work/free gap +0.0178
#     simu_sigma_aniso 0.2: Rfree 0.1462, work/free gap +0.0092
#
# 0.2 was the Rfree optimum on all three test datasets. Tightening to 0.1 halves
# the remaining gap again for ~0.002 in Rfree, if a tighter gap is worth that.
ADP_ANISO_SIGMA_LADDER = (1.0, 0.5, 0.2, 0.1)

# Macrocycles per trial during the search. Two reproduces the ranking that five
# gives, at ~40% of the cost.
ADP_SEARCH_CYCLES = 2

# Tolerance for the one-standard-error rule below. Measured by paired bootstrap
# on hewl (resample the free set, recompute both Rfree values on the *same*
# reflections): sigma = 0.0008 on the difference between two ladder rungs.
# Rounded up to 0.001.
ADP_SEARCH_RFREE_SE = 0.001

# LBFGS max_iter per resolution cutoff during rigid-body refinement.
#
# torchref's CLI default is 30, and its docstring warns that under-converges
# ("9RTS needs >= 100; raise it for production"). Measured on cxidb_81_small
# ep100 rather than assumed -- it is converged by 30 here:
#
#     iter/cutoff    10      30      50     100     200
#     Rwork       0.2010  0.2009  0.2009  0.2009  0.2009
#     Rfree       0.2167  0.2163  0.2164  0.2164  0.2164
#     shift rmsd   0.058   0.060   0.063   0.063   0.062
#
# 30 through 200 agree to 0.0001 in Rfree, i.e. noise, while 100 costs 51.5 s
# against 27.2 s for 30 (rigid-body step alone, CPU, measured sequentially).
# Doubling the cost buys nothing measurable, so 30 it is.
#
# The docstring's warning is about models needing a *large* correction; these
# start 0.06 A out. A molecular-replacement solution could genuinely need more
# iterations, which is what RIGID_BODY_LARGE_SHIFT below exists to flag.
RIGID_BODY_ITER = 30

# Shift (A, rmsd) above which a rigid-body step counts as a large correction.
#
# Convergence was verified only for the ~0.06 A corrections the benchmark models
# need. A bigger correction is the regime torchref's docstring warns about, and
# nothing else in the output would reveal an under-converged step, so say so.
RIGID_BODY_LARGE_SHIFT = 0.5

# Columns rsbooster's peak_report emits, preserved so peaks.csv stays a
# drop-in for anything downstream (abismal-benchmarks/scripts/plot_progress.py
# reads residue/seqid/peakz).
PEAK_COLUMNS = [
    "chain", "seqid", "residue", "name", "dist", "peak", "peakz", "score",
    "scorez", "cenx", "ceny", "cenz", "coordx", "coordy", "coordz",
]

# Z-score cutoff for anomalous peak finding, matching AnomalousPeakFinder.
Z_SCORE_CUTOFF = 5.0

# FFT oversampling used when transforming map coefficients to a real-space grid.
#
# `peakz` is a max over grid nodes, so it underestimates the continuous maximum
# by more the coarser the grid. Measured on cxidb_81_small epoch 100 (grid
# spacing in A, peak z-score at the Zn site):
#
#     sample_rate   3      4      5      6      7      8
#     spacing    0.65   0.49   0.39   0.33   0.29   0.24
#     ZN317     23.50  23.08  24.26  24.29  24.14  24.73
#
# It does not converge -- the estimator is biased low and the bias only shrinks,
# jittering +-0.3 with where nodes fall (note 4 reads *below* 3). 5 buys ~0.5
# sigma over 3 and escapes the coarse-grid dips, for 0.39 s against 0.10 s per
# epoch, so it is where we stop. Because it is not converged, the value must
# stay FIXED and identical on both sides of any comparison; see
# find_anomalous_peaks for the matching dmin requirement. Removing the bias
# rather than shrinking it needs subpixel refinement in rsbooster.
#
# SAMPLE_RATE also sets the *resolving* power, because PEAK_MIN_DISTANCE is in
# voxels: 3 voxels is 3 * spacing Angstroms, and two peaks closer than that
# collapse to one. hewl has four disulfides 2.02-2.05 A apart plus two
# methionines, so 10 sites is the correct answer:
#
#     sample_rate      3      4      5      6
#     spacing (A)  0.551  0.413  0.331  0.275
#     3 voxels (A)  1.65   1.24   0.99   0.83
#     peaks found      7      9     10     10
#
# 5.0 is the first rung that resolves every disulfide; 3.0 would silently merge
# three of them. Lowering SAMPLE_RATE therefore costs real sites, and raising it
# past 5 buys nothing on either axis.
#
# Note sample_rate is d_min/spacing, so critical (Nyquist) sampling is 2.0 and
# 5.0 is 2.3x Nyquist after grid rounding, not 5x.
SAMPLE_RATE = 5.0

# Markers delimiting the machine-readable summary at the end of stdout. Keep
# these in sync with the parser in abismal-benchmarks/scripts/plot_progress.py.
SUMMARY_BEGIN = "=== torchref summary ==="
SUMMARY_END = "=== end torchref summary ==="
PEAKS_BEGIN = "--- peaks.csv ---"
PEAKS_END = "--- end peaks.csv ---"

# The only columns handed to torchref for anomalous data. See prepare_data.
AMPLITUDE_COLUMNS = ("F(+)", "SIGF(+)", "F(-)", "SIGF(-)")

# R-free column names, in the same priority order torchref's reader uses. The
# first entry is what the joined column gets renamed to.
R_FREE_FLAG_NAMES = (
    "R-free-flags", "RFREE", "FreeR_flag", "FREE", "R-free", "Rfree",
    "FREER", "FREE_FLAG",
)


def _is_anomalous(ds):
    """Detect anomalous data from the presence of Friedel-split columns."""
    return "F(+)" in ds.columns


def resolve_adp_mode(adp_mode, pdb_path):
    """Pick isotropic vs anisotropic ADPs, auto-detecting from the model.

    A model deposited with ANISOU records was refined anisotropically, and
    reproducing that needs the anisotropic parameterization -- refining a
    scalar B against it discards the tensor and costs real R. Measured on hewl:
    anisotropic with a fitted restraint weight reaches Rfree 0.1468 against
    0.1553 isotropic, consistently across three datasets.
    """
    if adp_mode != "auto":
        return adp_mode
    n_anisou = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ANISOU"):
                n_anisou += 1
    mode = "anisotropic" if n_anisou else "isotropic"
    print(
        f"adp-mode auto -> {mode} ({n_anisou} ANISOU records in the model)",
        flush=True,
    )
    return mode


def search_aniso_sigma(build_refinement, out_dir, ladder=ADP_ANISO_SIGMA_LADDER):
    """Fit SIMU's deviatoric sigma by cross-validation on Rfree.

    Refines at each candidate sigma and keeps the one with the lowest Rfree --
    the same criterion phenix's own weight optimisation uses. See
    ADP_ANISO_SIGMA_LADDER for why this dial rather than the `adp` group weight.

    Rfree is what is being predicted, so it is what to select on. A work/free
    gap of ~0.009 survives at the optimum: that is the honest residual of 6,770
    parameters against ~650 free reflections, and closing it further costs Rfree.

    Refit every epoch against that epoch's own merged data, deliberately. An
    earlier version fitted once and cached: on hewl that fitted at epoch 1, when
    abismal has not converged, merging error dominates and the whole ladder sits
    within 0.0015 in Rfree -- no signal, so the argmin was noise. It picked 0.5
    and froze it, and by epoch 8 that was far too loose (gap +0.019 against
    +0.010 for a refit). The data improves every epoch, so the fit has to track
    it.
    """
    import torch

    print(
        f"fitting SIMU deviatoric sigma over {list(ladder)} "
        f"({ADP_SEARCH_CYCLES} macrocycles per trial)...",
        flush=True,
    )
    results = []
    for w in ladder:
        ref = build_refinement(w)
        reset_b_factors(ref, torch)
        ref.get_scales()
        # The trial is only meaningful if it actually ran at sigma=w. Without
        # this the ladder can silently collapse to N identical runs at the
        # default and still report a confident fitted value.
        spec = {"simu": {"simu_sigma_aniso": float(w)}}
        for _ in range(ADP_SEARCH_CYCLES):
            apply_adp_restraints(ref, spec, f"sigma={w:g} trial")
            ref.refine_scaler()
            ref.refine_adp()
        rw, rf = ref.get_rfactor()
        results.append((rf, w, rw))
        print(f"    sigma={w:<6g} Rwork={rw:.4f} Rfree={rf:.4f} gap={rf-rw:+.4f}", flush=True)

    # One-standard-error rule: of the rungs whose Rfree is statistically
    # indistinguishable from the best, take the most restrained (smallest
    # sigma). Selecting on the raw argmin does not work here -- a paired
    # bootstrap on hewl puts the difference between neighbouring rungs at ~0.4
    # sigma, with 64% of resamples favouring the winner, so the argmin is mostly
    # noise. The work/free gap, by contrast, varies monotonically and far
    # outside noise across the same ladder (0.026 -> 0.004), so breaking ties
    # toward more restraint is both statistically standard and the thing that
    # actually controls overfitting. At epoch 1 on hewl the raw argmin picked
    # 0.5 and left a +0.019 gap; this rule picks 0.2 from the same numbers.
    rf_best = min(r[0] for r in results)
    tied = [r for r in results if r[0] <= rf_best + ADP_SEARCH_RFREE_SE]
    rf, best, rw = min(tied, key=lambda r: r[1])

    note = "" if best == min(r[1] for r in results if r[0] == rf_best) else \
        f", 1-SE rule over {len(tied)} tied rung(s)"
    print(
        f"adp aniso sigma fitted: {best:g} "
        f"(Rfree={rf:.4f}, gap={rf-rw:+.4f}{note})",
        flush=True,
    )
    if best == ladder[-1]:
        print(
            f"    NOTE: the most restrained rung won, so the true optimum may "
            f"lie below sigma={ladder[-1]:g}.",
            flush=True,
        )
    return best


def adp_restraint_spec(adp_mode, aniso_sigma):
    """The ADP restraint parameters this worker owns, as {component: {attr: value}}.

    One place that says what we intend, so `apply_adp_restraints` can re-assert
    it after anything that rebuilds the targets. Empty for isotropic runs, which
    keep torchref's defaults on every channel.
    """
    if adp_mode == "anisotropic" and aniso_sigma is not None:
        # Only the anisotropy channel is touched; the `adp` group weight and the
        # magnitude sigmas keep torchref's defaults.
        return {"simu": {"simu_sigma_aniso": float(aniso_sigma)}}
    return {}


def apply_adp_restraints(ref, spec, where):
    """(Re)apply ADP component restraint parameters and verify they took.

    torchref keeps these on the ADP *target objects*, and
    `Refinement._init_targets` rebuilds those from scratch passing no restraint
    parameters -- once per resolution cutoff inside `refine_rigid_body`, and
    again on the ensemble and `create_from_state_dict` paths. A rebuild
    therefore resets whatever we set back to the component defaults
    (`simu_sigma` 2.0, `simu_sigma_aniso` 1.0), silently: no error, no warning,
    and the run continues at a restraint weight nobody chose.

    So the worker holds its own copy and re-asserts it. The call is idempotent
    and costs nothing measurable, so it goes at the top of every macrocycle
    rather than only where a rebuild is known to happen today -- that way
    enabling rigid body later cannot quietly invalidate a fitted sigma.

    The read-back assert is the point. Without it this helper would fail the
    same silent way the bug does if torchref ever renames an attribute.

    Group *weights* need no propagation: `ref.weighting` lives on the Refinement,
    not on the targets, and survives rigid body and macrocycles untouched
    (verified 2026-08-20). Only these component sigmas are at risk.

    Fixed upstream on kmdalton/TorchRef branch
    `fix/adp-restraint-config-lost-on-rebuild`, which adds an `adp_restraints=`
    constructor argument that reapplies on every rebuild. Delete this helper and
    pass that instead once it lands.
    """
    if not spec:
        return
    for component, attrs in spec.items():
        target = ref.adp_target[component]
        for attr, value in attrs.items():
            setattr(target, attr, value)
            got = getattr(target, attr)
            if abs(float(got) - float(value)) > 1e-5:
                raise RuntimeError(
                    f"adp/{component}.{attr} did not stick at {where}: "
                    f"set {value:g}, read back {float(got):g}. A target rebuild "
                    "happened after this call, or the attribute was renamed."
                )


def restraint_audit(ref, spec):
    """One-line summary of what the restraints and weights actually are.

    Printed per macrocycle so a run's log carries evidence of the values it used
    rather than the values it was asked for -- the distinction this whole helper
    exists to preserve.
    """
    bits = []
    for component, attrs in (spec or {}).items():
        for attr in attrs:
            bits.append(f"{component}.{attr}={float(getattr(ref.adp_target[component], attr)):g}")
    try:
        weights = ref.weighting(ref.complete_loss_state())
        bits.append(f"w[adp]={float(weights.get('adp', float('nan'))):g}")
    except Exception:
        pass
    return "  ".join(bits)


def run_rigid_body(ref, spec, iterations=RIGID_BODY_ITER):
    """Per-chain rigid-body refinement, then put the scaler and restraints back.

    Three things have to follow the rigid-body step, and all three are easy to
    miss because omitting any of them fails quietly rather than loudly:

    1. `get_scales()`. The scaler was fitted against the old coordinates, and
       the bulk-solvent mask was built from them. Rebuilding is also what makes
       the stale `Scaler._f_sol_raw` cache irrelevant -- torchref's macrocycle
       loop calls `solvent.update_solvent()` without clearing that cache, so the
       rebuilt mask would otherwise never reach F_calc. Inert while coordinates
       are frozen; live the moment rigid body moves them.
    2. Re-assert the ADP restraints. `refine_rigid_body` rebuilds the ADP
       targets once per resolution cutoff, resetting component sigmas to their
       defaults. See apply_adp_restraints.
    3. Report the shift. A rigid-body step that moved nothing means the model
       was already in register, which is worth knowing; a large shift is worth
       knowing for the opposite reason.

    Runs before the macrocycle loop, matching the benchmark's phenix protocol
    (`strategy = *rigid_body *individual_adp`) -- and phenix likewise runs it
    once, up front, not per macrocycle.
    """
    import torch

    with torch.no_grad():
        before = ref.model.xyz().detach().clone()

    rw0, rf0 = ref.get_rfactor()
    ref.refine_rigid_body(iterations_per_step=iterations)

    # Scaler and solvent mask were fitted against the pre-shift coordinates.
    ref.get_scales()
    # And the ADP targets were just rebuilt out from under us.
    apply_adp_restraints(ref, spec, "rigid body")

    with torch.no_grad():
        shift = (ref.model.xyz().detach() - before).norm(dim=-1)
        rmsd = float(shift.pow(2).mean().sqrt())
        dmax = float(shift.max())
    rw, rf = ref.get_rfactor()
    print(
        f"Rigid body ({iterations} iter/cutoff): "
        f"Rwork {rw0:.4f}->{rw:.4f}  Rfree {rf0:.4f}->{rf:.4f}  "
        f"shift rmsd={rmsd:.3f} A max={dmax:.3f} A",
        flush=True,
    )
    if rmsd > RIGID_BODY_LARGE_SHIFT and iterations < 100:
        print(
            f"    NOTE: {rmsd:.3f} A is a large correction, beyond the regime "
            f"where {RIGID_BODY_ITER} iterations were shown to converge. "
            "Consider --rigid-body-iter 100.",
            flush=True,
        )


def reset_b_factors(ref, torch):
    """Flatten every refinable B to RESET_B and invalidate the SF cache."""
    from torchref.model.parameter_wrappers import PositiveMixedTensor

    adp = ref.model.adp
    ref.model.adp = PositiveMixedTensor(
        torch.full_like(adp().detach(), RESET_B),
        refinable_mask=adp.refinable_mask,
        name="adp",
    )
    ref.model.reset_cache()


def _pick_free_value(values, counts, r_free_mtz):
    """Infer which flag value marks the free set: the minority of two values.

    Only safe for the two-value case. Multi-bin flag sets (phenix/CCP4 write
    0..19 and nominate one bin as the test set) are ambiguous by construction,
    so those must say which value they mean.
    """
    if len(values) == 2:
        return int(values[int(counts[0] > counts[1])])
    listing = ", ".join(f"{v}: {c}" for v, c in zip(values, counts))
    raise ValueError(
        f"Cannot infer the free-set flag value in {r_free_mtz}: found "
        f"{len(values)} distinct values ({listing}). Multi-bin flag sets are "
        "ambiguous -- pass --r-free-value to say which one is the test set."
    )


def join_r_free_flags(ds, r_free_mtz, r_free_value=None):
    """Attach an external R-free set to the reflections torchref will refine on.

    abismal's per-epoch MTZ carries no R-free flags, so without this torchref
    generates a fresh random 2% set on every load -- which makes Rfree noisy
    from epoch to epoch and not comparable to a phenix run using a fixed set.

    The flags are matched on Miller index, so the file may cover more
    reflections than the merged data; unmatched reflections are left unflagged
    and torchref puts them in the working set.

    ``r_free_value`` names the integer that marks a *free* reflection -- the
    equivalent of the phenix GUI's test-flag value. When omitted it is inferred
    for two-valued flag sets. Flags are then rewritten to torchref's own
    convention (0 = free, 1 = work), which sidesteps the guess-and-flip
    heuristic in its MTZ reader: that heuristic looks at what fraction of the
    values are 0 and inverts if it exceeds half, so it can reach a different
    verdict on a merged subset than on the full flag file.
    """
    import numpy as np
    import reciprocalspaceship as rs

    flags = rs.read_mtz(str(r_free_mtz))
    key = next((k for k in R_FREE_FLAG_NAMES if k in flags.columns), None)
    if key is None:
        raise ValueError(
            f"No R-free column in {r_free_mtz}. Looked for "
            f"{', '.join(R_FREE_FLAG_NAMES)}; found {list(flags.columns)}."
        )

    flags = flags[[key]]
    flags = flags.loc[~flags.index.duplicated()]

    raw = flags[key].to_numpy("float64")
    present = raw[np.isfinite(raw)].astype("int32")
    values, counts = np.unique(present, return_counts=True)

    if r_free_value is None:
        free_value = _pick_free_value(values, counts, r_free_mtz)
        how = "inferred"
    else:
        free_value = int(r_free_value)
        how = "given"
        if free_value not in values.tolist():
            raise ValueError(
                f"--r-free-value {free_value} does not occur in {r_free_mtz}; "
                f"it holds {', '.join(str(v) for v in values.tolist())}."
            )

    # Rewrite to torchref's convention: 0 marks the test set, 1 the work set.
    flags[R_FREE_FLAG_NAMES[0]] = np.where(
        raw == free_value, 0, 1
    ).astype("int32")
    flags = flags[[R_FREE_FLAG_NAMES[0]]].astype({R_FREE_FLAG_NAMES[0]: "MTZInt"})

    out = ds.join(flags, how="left")
    joined = out[R_FREE_FLAG_NAMES[0]].to_numpy("float64")
    matched = int(np.isfinite(joined).sum())
    if matched == 0:
        raise ValueError(
            f"No reflection in {r_free_mtz} matched the merged data. The flags "
            "are probably on a different asymmetric unit or cell."
        )

    n_free = int((joined == 0).sum())
    free_pct = 100.0 * n_free / matched
    print(
        f"joined R-free flags from {r_free_mtz} (column {key!r}, free value "
        f"{free_value} [{how}]): {matched}/{len(out)} reflections matched, "
        f"{n_free} free ({free_pct:.1f}%)",
        flush=True,
    )
    if free_pct > 50.0:
        print(
            f"WARNING: {free_pct:.1f}% of matched reflections are in the free "
            "set. That is almost certainly the wrong --r-free-value, and "
            "torchref's reader will invert it back.",
            flush=True,
        )
    return out


def prepare_data(mtz_path, pdb_path, out_dir, r_free_mtz=None,
                 r_free_value=None):
    """Read the abismal MTZ, prepare it for torchref, and write a temp copy.

    F(+)/F(-) are passed through untouched: torchref keeps a signed
    ``hkl_anomalous`` index and refines against both Friedel mates, so there is
    nothing to merge here. The data cell is stamped to the model cell so
    refinement, scaling, the output PDB and the map MTZ all share one cell.

    Returns ``(input_mtz_path, anomalous)``.
    """
    import gemmi
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(mtz_path))
    anomalous = _is_anomalous(ds)
    print(
        f"{len(ds)} reflections, anomalous={anomalous}"
        + (" (both Friedel mates go to refinement)" if anomalous else ""),
        flush=True,
    )

    # Keep ONLY the amplitude columns. abismal also writes I(+)/I(-) and
    # E(+)/E(-); left in place, torchref's reader prefers the intensities and
    # re-derives F from them via French-Wilson, silently ignoring the structure
    # factors we actually want to refine against. The benchmark's phenix config
    # disables exactly this (`french_wilson_scale = False`, amplitudes selected).
    #
    # This is a whitelist rather than a blacklist on purpose: the previous
    # blacklist tested `isinstance(dtype, rs.IntensityDtype)`, which is False
    # for the `FriedelIntensity` dtype of I(+)/I(-), so it dropped nothing at
    # all on anomalous data -- the case that matters.
    keep = [c for c in (AMPLITUDE_COLUMNS if anomalous else ("F", "SIGF"))
            if c in ds.columns]
    if keep:
        dropped = [c for c in ds.columns if c not in keep]
        if dropped:
            print(f"keeping {keep}, dropping {dropped}", flush=True)
        ds = ds[keep]
    else:
        print(
            "WARNING: no amplitude columns found; leaving the data as-is, so "
            "torchref will derive amplitudes from whatever it does find.",
            flush=True,
        )

    if r_free_mtz is not None:
        ds = join_r_free_flags(ds, r_free_mtz, r_free_value=r_free_value)

    ds.cell = gemmi.read_structure(str(pdb_path)).cell

    # Named to end in 'data.mtz' so the GUI result-viewer's exclusion glob skips
    # it and only picks up the refined output mtz.
    input_mtz = Path(out_dir) / "input_data.mtz"
    ds.write_mtz(str(input_mtz))
    return str(input_mtz), anomalous


# Minimum separation between reported peaks, in grid voxels.
#
# peak_local_max keeps only the largest maximum within this radius, so it sets
# the resolving power. 3 voxels is ~1.2 A at SAMPLE_RATE 5 -- fine enough to
# split hewl's disulfide sulfur pairs (2.02-2.05 A apart), which the flood fill
# merged into one blob apiece. Must stay below the closest pair worth
# resolving: at >= 6 voxels HOH1002 is absorbed into ZN317 3.31 A away.
PEAK_MIN_DISTANCE = 3


def periodic_peak_indices(arr, min_distance, threshold):
    """Local maxima of a periodic map, as indices into ``arr``.

    Pad with wrapped copies so every voxel inside the real cell sees its true
    periodic neighbourhood, run peak_local_max, then CROP back to the real
    extent. Cropping rather than folding the padding detections back with a
    modulo is what makes this exact: a padding detection is by construction a
    duplicate of one inside the cell, so discarding it loses nothing and cannot
    double-count. Verified exactly roll-invariant over 9 shifts; folding was not.

    `exclude_border=False` is required -- the default silently drops maxima
    within `min_distance` of the array edge, which for a periodic map is not a
    border at all.
    """
    from skimage.feature import peak_local_max

    pad = int(min_distance)
    padded = np.pad(arr, pad, mode="wrap")
    pk = peak_local_max(
        padded, min_distance=min_distance, threshold_abs=threshold,
        exclude_border=False,
    )
    n = np.array(arr.shape)
    inside = np.all((pk >= pad) & (pk < pad + n), axis=1)
    return pk[inside] - pad


def dedup_symmetry(idx, shape, spacegroup):
    """Keep one representative per symmetry-equivalent group of peaks.

    peak_local_max sees a P1 map and reports all N copies of every peak; gemmi's
    flood fill dedupes internally with an ASU mask. Canonicalise rather than
    mask: map each peak through every operator and key on the lexicographically
    smallest grid index. Masking would drop a peak whose maximum happens to sit
    on the far side of an ASU boundary; canonicalising cannot.

    `idx` must be sorted by descending height, so each group keeps its strongest
    member.
    """
    if len(idx) == 0:
        return idx
    ops = list(spacegroup.operations())
    n = np.array(shape, dtype=float)
    ni = np.array(shape)
    seen, keep = set(), []
    for i, frac in enumerate(idx / n):
        key = min(
            tuple(np.mod(np.round(np.mod(op.apply_to_xyz(list(frac)), 1.0) * n).astype(int), ni))
            for op in ops
        )
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return idx[keep]


def _peak_region(arr, index, radius, threshold):
    """Offsets and values of the above-threshold voxels around one peak.

    Offsets are relative to the peak, so unit-cell wrapping needs no special
    case. Used for the integrated score and the intensity-weighted centroid,
    keeping both comparable to what the flood fill reported.
    """
    n = np.array(arr.shape)
    rng = [np.arange(-radius, radius + 1)] * 3
    off = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, 3)
    off = off[(off ** 2).sum(axis=1) <= radius ** 2]
    vals = arr[tuple(np.mod(index + off, n).T)]
    keep = vals >= threshold
    return off[keep], vals[keep]


def peaks_by_local_max(structure, grid, z_score_cutoff,
                       min_distance=PEAK_MIN_DISTANCE, distance_cutoff=4.0):
    """Anomalous peaks via skimage's peak_local_max, in peak_report's schema.

    Replaces gemmi flood fill (rsbooster's ``peak_report``). A local maximum is
    one voxel larger than its neighbours, so it has no volume to lose as the
    threshold rises -- which removes the failure mode flood fill has, where a
    blob evaporates under gemmi's hard 3-voxel floor and a genuine site vanishes
    from the table. Results are identical from 0 to 5 sigma, so the detection
    threshold is simply the reporting cutoff; no margin heuristic is needed.

    It also resolves peaks the flood fill merges: on hewl this reports both
    sulfurs of all four disulfides (10 peaks) where flood fill reported one per
    disulfide (6).
    """
    import gemmi
    import pandas as pd

    arr = np.array(grid, copy=False)
    mu, sd = float(arr.mean()), float(arr.std())
    threshold = mu + z_score_cutoff * sd

    idx = periodic_peak_indices(arr, min_distance, threshold)
    if len(idx) == 0:
        return pd.DataFrame(columns=PEAK_COLUMNS)
    idx = idx[np.argsort(-arr[tuple(idx.T)])]
    idx = dedup_symmetry(idx, arr.shape, grid.spacegroup)

    model, cell = structure[0], structure.cell
    ns = gemmi.NeighborSearch(model, cell, distance_cutoff).populate()
    voxel_volume = cell.volume / arr.size
    n = np.array(arr.shape, dtype=float)

    rows = []
    for i in idx:
        peak_value = float(arr[tuple(i)])
        off, vals = _peak_region(arr, i, min_distance, threshold)
        score = float(vals.sum()) * voxel_volume
        # Intensity-weighted centroid, in offsets from the peak so that wrapping
        # is automatic. This is the flood fill's position convention; the peak
        # voxel alone sits systematically further from the atom.
        centroid_idx = i + (off * vals[:, None]).sum(axis=0) / vals.sum()
        centroid = cell.orthogonalize(gemmi.Fractional(*(centroid_idx / n)))

        best = None
        for mark in ns.find_atoms(centroid):
            cra = mark.to_cra(model)
            dist = cell.find_nearest_pbc_image(centroid, cra.atom.pos, mark.image_idx).dist()
            if best is None or dist < best[0]:
                best = (dist, cra)
        if best is None or best[0] > distance_cutoff:
            continue
        dist, cra = best
        rows.append({
            "chain": cra.chain.name,
            "seqid": cra.residue.seqid.num,
            "residue": cra.residue.name,
            "name": "" if cra.residue.is_water() or len(cra.residue) == 1 else cra.atom.name,
            "dist": dist,
            "peak": peak_value,
            "peakz": (peak_value - mu) / sd,
            "score": score,
            "scorez": score / sd,
            "cenx": centroid.x, "ceny": centroid.y, "cenz": centroid.z,
            "coordx": cra.atom.pos.x, "coordy": cra.atom.pos.y, "coordz": cra.atom.pos.z,
        })
    return pd.DataFrame(rows, columns=PEAK_COLUMNS)


def find_anomalous_peaks(refined_mtz, pdb_file, out_csv,
                         z_score_cutoff=Z_SCORE_CUTOFF, dmin=None):
    """Search torchref's anomalous difference map for peaks near the model.

    ANOM/PANOM come straight out of ``write_out_mtz`` on anomalous data, so the
    map is built from the anomalously refined model rather than reconstructed
    here.

    `dmin` fixes the FFT grid. gemmi has no dmin argument -- it sizes the grid
    from the largest Miller index *present as a row*, NaN values included -- so
    the cut has to happen before `transform_f_phi_to_map`. Reflections with a
    NaN coefficient contribute nothing to the map, so dropping them changes only
    the grid, never the density.

    This matters because `peakz` is a max over grid nodes, which underestimates
    the continuous maximum by more the coarser the grid. phenix pads
    `refine_001.mtz` out to 1.0 A with all-NaN ANOM, so gemmi hands it a 0.38 A
    grid where the same coefficients at abismal's true 1.8 A limit get 0.65 A --
    worth ~1 sigma on every peak, purely from sampling. Peak heights are only
    comparable on a common grid, so pin both sides to the same dmin.
    """
    import gemmi
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(refined_mtz))
    missing = [c for c in ("ANOM", "PANOM") if c not in ds.columns]
    if missing:
        print(
            f"refined mtz has no {'/'.join(missing)} column; skipping peak "
            f"finding (columns: {', '.join(ds.columns)})",
            flush=True,
        )
        return None

    if dmin is not None:
        n_before = len(ds)
        ds = ds.compute_dHKL()
        ds = ds.loc[ds["dHKL"] >= float(dmin)]
        print(
            f"peak finding at dmin={float(dmin):.2f} A: kept {len(ds)}/{n_before} "
            "reflections (fixes the FFT grid)",
            flush=True,
        )

    structure = gemmi.read_pdb(str(pdb_file))
    mtz = ds[["ANOM", "PANOM"]].to_gemmi()
    grid = mtz.transform_f_phi_to_map("ANOM", "PANOM", sample_rate=SAMPLE_RATE)
    print(
        f"anomalous map grid {tuple(grid.shape)} "
        f"({structure.cell.a / grid.shape[0]:.2f} A spacing, "
        f"sample_rate={SAMPLE_RATE})",
        flush=True,
    )

    report = peaks_by_local_max(structure, grid, z_score_cutoff)
    print(
        f"peak_local_max (min_distance={PEAK_MIN_DISTANCE} voxels): "
        f"{len(report)} peaks at or above {z_score_cutoff:g} sigma",
        flush=True,
    )
    report.to_csv(str(out_csv), index=False)
    print(
        f"found {len(report)} anomalous peaks above {z_score_cutoff} sigma -> {out_csv}",
        flush=True,
    )
    return report


def run(mtz_path, pdb_path, out_dir, device="cpu", macro_cycles=MACRO_CYCLES,
        z_score_cutoff=Z_SCORE_CUTOFF, r_free_mtz=None, r_free_value=None,
        wavelength=None, adp_mode="auto", adp_aniso_sigma="auto",
        peak_dmin=None, rigid_body=True, rigid_body_iter=RIGID_BODY_ITER):
    import torch
    from torchref import LBFGSRefinement

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_mtz, anomalous = prepare_data(
        mtz_path, pdb_path, out_dir, r_free_mtz=r_free_mtz,
        r_free_value=r_free_value,
    )
    print(f"anomalous={anomalous}  device={device}", flush=True)

    adp_mode = resolve_adp_mode(adp_mode, pdb_path)
    dev = torch.device(device)
    # Only pass a wavelength when one was supplied, so torchref's own default
    # (1.0) stays the single source of truth otherwise. It drives the f'/f''
    # correction; on hewl it moved R by <0.001, but it matters more for heavier
    # anomalous scatterers and for wavelengths near an edge.
    kwargs = {} if wavelength is None else {"wavelength": float(wavelength)}

    def build(aniso_sigma):
        r = LBFGSRefinement(
            data_file=input_mtz,
            pdb=str(pdb_path),
            device=dev,
            adp_mode=adp_mode,
            # Read's MLF sigma_A likelihood: variance eps*beta, conditional
            # mean alpha*|F_calc|. torchref's default and the state of the art.
            target_mode="ml",
            # The scale fit gets the same sigma_A likelihood, with the Luzzati
            # coupling pinned at 1 -- alpha is degenerate with the scale being
            # fitted, so torchref admits only {nll, ml_noalpha} here. The
            # default `nll` is a plain sigma-weighted Gaussian with no
            # model-error term at all, and its per-bin log_scale can collapse
            # in weak shells: on the phenix reference amplitudes that cost
            # 0.022 in Rfree (0.1771 -> 0.1547).
            scale_target="ml_noalpha",
            # Refine against abismal's structure factors, never a French-Wilson
            # re-derivation from intensities. prepare_data already withholds
            # the intensity columns; this makes the intent explicit and holds
            # even if a future abismal output sneaks one back in. Matches the
            # benchmark's phenix config (`french_wilson_scale = False`).
            french_wilson=False,
            **kwargs,
        )
        apply_adp_restraints(
            r, adp_restraint_spec(adp_mode, aniso_sigma), "construction"
        )
        return r

    if adp_mode != "anisotropic":
        sigma = None
        print("isotropic ADPs: no anisotropy restraint to fit", flush=True)
    elif adp_aniso_sigma == "auto":
        sigma = search_aniso_sigma(build, out_dir)
    else:
        sigma = float(adp_aniso_sigma)
        print(f"adp aniso sigma: {sigma:g} (fixed)", flush=True)

    ref = build(sigma)

    # Reset every atomic B-factor to a flat starting value and invalidate the
    # structure-factor cache so the new values take effect. See RESET_B for the
    # ADP KL singularity this sits on.
    reset_b_factors(ref, torch)

    rw0, rf0 = ref.get_rfactor()
    print(
        f"Initial (B reset to {RESET_B}): Rwork={rw0:.4f}  Rfree={rf0:.4f}",
        flush=True,
    )

    # Macrocycles of bulk-solvent + scaling followed by atomic ADP refinement.
    # This mirrors what LBFGSRefinement.refine() does per cycle, minus the
    # refine_xyz() step: the benchmark's phenix protocol also refines only
    # `rigid_body` + `individual_adp`, and letting torchref move coordinates
    # lowered Rwork while raising Rfree (0.1576/0.1761 against 0.1671/0.1628
    # on hewl) -- overfitting, not a better model.
    #
    # get_scales() is a *cold* start: it replaces log_scale and rebuilds the
    # solvent and anisotropy terms, discarding the scale the previous cycle
    # refined. torchref's own docs say to call it once at construction and use
    # refine_scaler() inside the loop.
    ref.get_scales()
    spec = adp_restraint_spec(adp_mode, sigma)

    # Rigid body first, on the flat-B model, so the macrocycles refine ADPs
    # against a model that is already in register. Deliberately not run inside
    # search_aniso_sigma's trials: the search ranks *restraint* sigmas, that
    # ranking is unaffected by a sub-Angstrom global shift, and paying for it
    # per rung would multiply the search cost by the ladder length.
    if rigid_body:
        run_rigid_body(ref, spec, rigid_body_iter)

    for cycle in range(macro_cycles):
        # Re-assert the restraint configuration before every cycle. Nothing on
        # today's path rebuilds the ADP targets, so this is currently a no-op --
        # it is here so that adding a step which does (rigid body is the one we
        # want next) cannot silently drop a fitted sigma back to the default.
        # See apply_adp_restraints.
        apply_adp_restraints(ref, spec, f"cycle {cycle + 1}")
        if getattr(ref.scaler, "solvent", None) is not None:
            ref.scaler.solvent.update_solvent()
        ref.refine_scaler()
        ref.refine_adp()
        rw, rf = ref.get_rfactor()
        audit = restraint_audit(ref, spec)
        print(
            f"Cycle {cycle + 1:2d}/{macro_cycles}: Rwork={rw:.4f}  Rfree={rf:.4f}"
            + (f"  [{audit}]" if audit else ""),
            flush=True,
        )

    refined_pdb = out_dir / "refined.pdb"
    refined_mtz = out_dir / "refined.mtz"
    ref.write_out_pdb(str(refined_pdb))
    # On anomalous data this also writes F-obs(+/-), F-model(+/-) and the
    # ANOM/PANOM difference map coefficients that peak finding runs on.
    ref.write_out_mtz(str(refined_mtz))
    print(f"wrote refined.pdb and refined.mtz to {out_dir}", flush=True)

    peaks_csv = None
    if anomalous:
        out_csv = out_dir / "peaks.csv"
        if find_anomalous_peaks(
            refined_mtz, refined_pdb, out_csv, z_score_cutoff, dmin=peak_dmin
        ) is not None:
            peaks_csv = out_csv

    rw, rf = ref.get_rfactor()
    print_summary(rw, rf, peaks_csv, adp_mode=adp_mode, adp_aniso_sigma=sigma)


def print_summary(rwork, rfree, peaks_csv=None, adp_mode=None, adp_aniso_sigma=None):
    """Print the final R-factors and peak table at the end of stdout.

    Delimited so it can be recovered from a finished run's stdout.txt without
    re-reading the MTZ -- these are torchref's own numbers, not a recomputation.
    The block is written last so a plain `tail` shows it.
    """
    print(f"\n{SUMMARY_BEGIN}", flush=True)
    print(f"Rwork={rwork:.4f}", flush=True)
    print(f"Rfree={rfree:.4f}", flush=True)
    if adp_mode is not None:
        print(f"adp_mode={adp_mode}", flush=True)
    if adp_aniso_sigma is not None:
        print(f"adp_aniso_sigma={adp_aniso_sigma:g}", flush=True)
    if peaks_csv is not None and Path(peaks_csv).exists():
        print(PEAKS_BEGIN, flush=True)
        print(Path(peaks_csv).read_text().strip(), flush=True)
        print(PEAKS_END, flush=True)
    else:
        print("no anomalous peaks (non-anomalous data)", flush=True)
    print(SUMMARY_END, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="torchref refinement worker")
    p.add_argument("--mtz", required=True, help="abismal per-epoch MTZ to refine against")
    p.add_argument("--pdb", required=True, help="starting model")
    p.add_argument("--out-dir", required=True, help="directory for refined outputs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--macro-cycles", type=int, default=MACRO_CYCLES)
    p.add_argument(
        "--z-score-cutoff",
        type=float,
        default=Z_SCORE_CUTOFF,
        help="z-score cutoff for anomalous peak finding (anomalous data only)",
    )
    p.add_argument(
        "--r-free-mtz",
        default=None,
        help="MTZ supplying a fixed R-free set. Without it torchref generates a "
             "fresh random set on every run, making Rfree incomparable across "
             "epochs.",
    )
    p.add_argument(
        "--r-free-value",
        type=int,
        default=None,
        help="Integer flag value marking a FREE reflection in --r-free-mtz "
             "(phenix's test-flag value). Inferred for two-valued flag sets; "
             "required for multi-bin ones.",
    )
    p.add_argument(
        "--adp-mode",
        default="auto",
        choices=["auto", "isotropic", "anisotropic"],
        help="ADP parametrization. 'auto' (default) refines anisotropically "
             "when the model carries ANISOU records, isotropically otherwise.",
    )
    p.add_argument(
        "--adp-aniso-sigma",
        default="auto",
        help="Sigma on SIMU's deviatoric (anisotropy) channel, the dial that "
             "regularizes tensor shape. 'auto' (default) fits it by minimising "
             "Rfree over a coarse ladder, refit each epoch against that "
             "epoch's own data; or give a number to pin it. Isotropic runs "
             "ignore it.",
    )
    p.add_argument(
        "--no-rigid-body",
        dest="rigid_body",
        action="store_false",
        help="Skip the per-chain rigid-body step that runs before the ADP "
             "macrocycles. On by default: it is 6 parameters per chain against "
             "tens of thousands of reflections, so it cannot overfit, and it "
             "improves both Rwork and Rfree whenever the starting model is not "
             "already in register with the data (worth ~0.004 on both for "
             "cxidb_81_small, ~0.000 on hewl).",
    )
    p.add_argument(
        "--rigid-body-iter",
        type=int,
        default=RIGID_BODY_ITER,
        help=f"LBFGS max_iter per resolution cutoff during rigid-body "
             f"refinement (default {RIGID_BODY_ITER}). torchref's own default "
             "of 30 under-converges by its docstring.",
    )
    p.add_argument(
        "--peak-dmin",
        type=float,
        default=None,
        help="High-resolution limit (Angstroms) for the anomalous difference "
             "map. Reflections beyond it are dropped before the FFT, which is "
             "the only way to pin the grid -- gemmi sizes it from the largest "
             "Miller index present, counting NaN-valued rows. Peak z-scores are "
             "a max over grid nodes and so are only comparable between programs "
             "on a common grid; set this to the same value on both sides. "
             "Omitted uses whatever the refined MTZ contains.",
    )
    p.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="Experimental wavelength in Angstroms, driving the f'/f'' "
             "anomalous correction. Omitted leaves torchref's default (1.0). "
             "0 disables anomalous refinement and forces a Friedel-merged read.",
    )
    args = p.parse_args(argv)
    run(
        args.mtz,
        args.pdb,
        args.out_dir,
        device=args.device,
        macro_cycles=args.macro_cycles,
        z_score_cutoff=args.z_score_cutoff,
        r_free_mtz=args.r_free_mtz,
        r_free_value=args.r_free_value,
        wavelength=args.wavelength,
        adp_mode=args.adp_mode,
        adp_aniso_sigma=args.adp_aniso_sigma,
        peak_dmin=args.peak_dmin,
        rigid_body=args.rigid_body,
        rigid_body_iter=args.rigid_body_iter,
    )


if __name__ == "__main__":
    main()
