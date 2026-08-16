# Matching phenix refinement quality in the torchref worker

Analysis date: 2026-08-15. Benchmark: `results/torchref/cxidb_81_small`, epoch 100
(abismal training converged and flat: Rwork/Rfree move <0.0002 from epoch 90 to 100).
Cross-checked on `results/torchref/cxidb_81` (full).

Baseline: torchref **0.2050 / 0.2206**, phenix **0.1973 / 0.2053**.

## 1. What the two runs actually did

Both refine **B-factors only** -- this is an apples-to-apples comparison.
Phenix's eff selects `rigid_body + individual_adp` (no `individual_sites`,
`ordered_solvent = False`); the worker deliberately skips `refine_xyz()`.
Measured coordinate RMSD from the 2tli starting model: torchref **0.0000 A**,
phenix 0.060 A (the rigid-body shift).

| | torchref | phenix |
|---|---|---|
| Coordinates | none | rigid body, 0.060 A |
| Bonded `<|Bi-Bj|>` | **4.21 A^2** | **2.18 A^2** (deposited 2tli: 2.66) |
| B range / sigma | 8.1-**92.9**, s=8.95 | 8.8-51.0, s=5.94 |
| Rfree - Rwork | 0.0156 | 0.0080 |
| Bulk solvent | one global `k_sol*exp(-B_sol s^2)` = 0.351 / 45.98 | **per-bin free k_mask** (0.33 -> 0.32 -> 0.155 -> 0.029 -> 0) |
| Scale bins | 10 | ~30 + smoothing |
| Map coefficients | unweighted `|2Fo-Fc|`, `|dF_anom|` | `2mFo-DFc`, `m*dF_anom` |

The doubled `<|Bi-Bj|>`, B running to 93 A^2, and the 2x larger Rfree-Rwork gap
are all one defect: **the ADP restraint is far too weak.**

## 2. Why: torchref has no target-weight inference at all

There is no `wxc_scale` / `wxu_scale` / `optimize_adp_weight` analogue. Three
hard-coded constants in `torchref/refinement/base_refinement.py:45-56`:

```python
DEFAULT_GROUP_WEIGHTS = {
    "xray": 1.0, "geometry": 0.2, "geometry/ramachandran": 0.0,
    "adp": 0.02, "adp/sigd": 1.0,
}
```

Picked by a 10x10 log grid search on an AlphaFold-start benchmark
(`paper/figure2_alphafold_start/analysis/submit_weight_grid.py`), not derived
per-structure. `ManualWeighting` is the only weighting scheme
(`weighting/static_weighting.py:43-45` ignores `LossState` entirely).
Gradient-norm matching was explicitly evaluated and **rejected as circular**
(commit `9371b7e`); R-free cross-validation is named there as the intended
replacement and was never implemented. `utils/gradnorm.py` exists but is
monitoring-only. torchref's own paper reports main-chain B RMSZ 1.56 vs phenix's
1.03 -- the same defect measured here.

Note `geometry/ramachandran = 0.0` **disables the Ramachandran restraint by default.**

What torchref does have instead: a properly free-set-cross-validated per-shell
sigma_A (`model_error_estimation/sigma_a.py`), fitted on `data.free.mask`
regardless of which subset the target scores.

## 3. Decomposition of the gap

Every model scored with `phenix.model_vs_data` on identical data and phenix's
own scaler -- this separates model quality from scaling.

| model | Rwork | Rfree |
|---|---|---|
| torchref as-run | 0.1976 | 0.2155 |
| + ADP weight 0.02 -> 0.2 | 0.2018 | 0.2111 |
| + rigid body | 0.1938 | 0.2103 |
| **+ both** | **0.1976** | **0.2063** |
| phenix | 0.1973 | 0.2053 |

Of the 0.0153 Rfree gap: **~0.009 weak ADP restraint, ~0.005 missing rigid body,
~0.005 torchref's scaler** (partly overlapping). With both model fixes the
torchref model is statistically indistinguishable from phenix's
(dRwork 0.0003, dRfree 0.0010).

### ADP weight ladder (torchref's own R; Rfree minimum is shallow, 0.1-0.5)

| adp weight | Rwork | Rfree | gap | B max | B sigma |
|---|---|---|---|---|---|
| 0.02 (default) | 0.2049 | 0.2205 | 0.0156 | 92.9 | 8.95 |
| 0.05 | 0.2060 | 0.2188 | 0.0128 | 78.6 | 7.80 |
| 0.1 | 0.2077 | 0.2178 | 0.0101 | 68.2 | 6.83 |
| **0.2** | 0.2103 | **0.2176** | 0.0072 | 58.3 | 5.79 |
| 0.5 | 0.2159 | 0.2194 | 0.0035 | 46.0 | 4.31 |
| 1.0 | 0.2217 | 0.2228 | 0.0011 | 37.8 | 3.17 |
| *(phenix)* | *0.1973* | *0.2053* | *0.0080* | *51.0* | *5.94* |

adp=0.2 lands almost exactly on phenix's B distribution and Rfree-Rwork gap.

### From-scratch R, no rescaling of any kind (2026-08-15)

The cleanest statement of the discrepancy. Both programs use
`R = sum|Fo-Fc| / sum Fo`; read each one's own columns and own flags.

**The reflection sets are the same.** torchref internal: 58,855 rows, 52,666
work, **5,739 free**, 450 masked (non-finite F). phenix: 58,401 rows, 52,662
work, **5,739 free**. Identical free sets; work differs by 4 (phenix outlier
removal). None of the gap comes from what is scored.

| source | Rwork | Rfree | reproduces? |
|---|---|---|---|
| phenix `F-obs-filtered` vs `F-model` | 0.1973 | 0.2053 | yes, exactly |
| torchref internal `Fo` vs `k*\|Fc\|` | 0.2050 | 0.2205 | yes, exactly |
| torchref from its written MTZ (ASU) | 0.2039 | 0.2194 | **no**, off 0.001 |
| torchref from its MTZ, (+)/(-) stacked | 0.2104 | 0.2251 | **no**, off 0.005 |

**torchref's MTZ does not reproduce torchref's own reported R.** It writes
31,697 ASU rows with Bijvoet-mean `F-obs`, but refines and scores over 58,855
stacked mates under a validity mask. Neither the ASU column nor a naive stack
recovers the internal number. Reporting inconsistency worth fixing upstream.

Use phenix's `F-obs-filtered`, not `F-obs`: the latter keeps abismal's input
scale while `F-model` is absolute, giving R ~ 865.

**The discrepancy in one number.** Ask each program what single `k` minimises
`||Fo - k*Fc||` on its own work set:

| | `<\|Fo\|>` | `<\|Fc\|>` | mean ratio | **LS optimum k** | R gain |
|---|---|---|---|---|---|
| phenix | 321.065 | 302.024 | 1.0630 | **1.0000** | +0.0000 |
| torchref | 0.34886 | 0.31669 | 1.1016 | **1.0295** | -0.0025 |

**phenix's F_model is already exactly at the least-squares optimum; torchref's
sits 3% below it.** Phenix fits scales by least squares against `|F_obs|` -- the
same criterion R measures -- so it lands on k=1 by construction. torchref fits
by ML with sigma_A weighting, a different objective that does not minimise R.
(Both mean ratios exceed 1; that is expected, since `|Fo|` carries noise that
adds in magnitude regardless of model.)

Net: `0.2050/0.2205` reported, `0.2025/0.2180` after removing the scale bias,
against phenix's `0.1973/0.2053`. The remainder is model quality -- the weak ADP
restraint and the missing rigid body.

### The residual scale bias

torchref's reported R is 0.006-0.008 worse than phenix's **on an identical
model**. A single global least-squares rescale by **k = 1.0294** drops
Rwork 0.2054 -> 0.2029 and Rfree 0.2134 -> 0.2109. Per-bin (30) LS reaches
0.2016 / 0.2100; more bins give nothing.

**Cause, verified 2026-08-15.** The `ml` body target is a Rice/folded normal
centred on `alpha*k*|F_calc|`, but `get_rfactor` computes R from
`_scaled_F_calc_full()` = `k*|F_calc|`, *without* alpha. Refinement drives
`k*|F_calc|` to whatever makes `alpha*k*|F_calc|` track `|F_obs|`, so the
quantity R uses is low by a factor of alpha.

Measured on the rb+adp0.2 model:

```
alpha   : min 0.9608  max 1.1592  mean 1.0303
sigma_A : min 0.9093  max 0.9715  mean 0.9421
LS rescale needed on k|Fc|       = 1.0295
```

`k_LS = 1.0295` matches `mean(alpha) = 1.0303` to three decimals -- the
mechanism, confirmed.

**alpha is GREATER than 1, not less.** An earlier version of this note said
"alpha < 1"; that conflated alpha with sigma_A. `alpha = sigma_A *
sqrt(Sigma_N/Sigma_P)`, and `Sigma_N/Sigma_P > 1` whenever the model explains
less scattering than the data contain -- the normal state for any real model
(torchref's own docstring says 1.02-1.6 on refined models). sigma_A is 0.942
here; alpha is 1.0303. The fix is unchanged (multiply by alpha), only the reason.

torchref already guards against this in the *scale* fit (`ml_noalpha`, see
`scaling/scaler_base.py:600-604`) but the bias re-enters through the ADP step,
which uses the alpha-free `ml` row.

**Full decomposition of the 0.0071 Rfree discrepancy** (torchref 0.2054/0.2134
vs phenix 0.1976/0.2063 on the identical rb+adp0.2 model, identical data, and
**identical free sets** -- both score the same 5,739 free reflections, verified):

| step | Rwork | Rfree | dRfree |
|---|---|---|---|
| torchref as reported | 0.2054 | 0.2134 | -- |
| x alpha (global LS, 1.0295) | 0.2029 | 0.2109 | 0.0025 |
| per-bin LS refit (30 bins) | 0.2016 | 0.2100 | 0.0009 |
| phenix, same model | 0.1976 | 0.2063 | 0.0037 *(unattributed)* |

**The remaining ~half is inference, not measurement.** Phenix recomputes F_calc
from the atoms and fits its own `k_isotropic` (per bin, smoothed),
`k_anisotropic` and `k_mask` (per bin, free). Two candidate causes:

1. **Fitting criterion.** Phenix fits scales by least squares against `|F_obs|`
   -- an R-like objective. torchref fits by ML NLL with sigma_A weighting, which
   downweights weak and high-resolution reflections and does not minimize R.
   Supported by: an LS refit of torchref's *own* scale always helps.
2. **Bulk-solvent shape.** torchref's global `k_sol*exp(-B_sol*s^2)`
   (0.351/45.98) against phenix's fitted per-bin k_mask is too low at low
   resolution (0.26 vs 0.31 at 6.2 A) and too high at high resolution (0.023 vs
   0.000 at 2.05 A).

Caveat on (2): torchref's own `setup_binwise_solvent_scale()` gained only 0.0007
Rfree, far less than the 0.0037 residual -- so it looks more like the criterion
than the parameterization. Not isolated component by component; do not quote the
attribution as established.

## 4. Anomalous peaks

**CORRECTED 2026-08-15.** An earlier version of this note claimed phenix
FOM-weights its anomalous map and torchref does not. **That is false.** Verified
in `mmtbx/map_tools.py:171-178`:

```python
# Special case #1: anomalous map
if(map_name_manager.anomalous):
  if(self.anom_diff is not None):
    # Formula from page 141 in "The Bijvoet-Difference Fourier Synthesis",
    # Jeffrey Roach, METHODS IN ENZYMOLOGY, VOL. 374
    return miller.array(miller_set = self.anom_diff,
                        data       = self.anom_diff.data()/(2j))
```

This is an **early return** -- it never reaches `self.mch`
(`map_calculation_helper`, where alpha/beta/FOM live) at line 207. Phenix's ANOM
is raw `dF_anom`, phase-transferred from the Friedel-averaged `F_model` and
divided by `2i`. **No figure of merit.** torchref's "Stored in the phenix
convention" comment (`io/datasets/reflection_data.py:2672-2674`) is accurate.

The bad claim came from assuming a general convention (m-weighted anomalous
difference Fouriers are common in *other* software) and then reading phenix's
tabulated `figures of merit ... mean = 0.86` as confirmation. That table is the
**phase** FOM from the ML/sigma_A estimate, used for 2mFo-DFc; it has nothing to
do with ANOM. Do not repeat this -- check `map_tools.py`.

So both programs use the same unweighted formula, and on a common grid they
agree: torchref without FOM at 0.65 A = **22.62** on ZN317, phenix at 0.65 A =
**22.65**. The entire apparent peak deficit was grid sampling (below).

FOM weighting remains available as a *speculative improvement over both*.
Computing m from torchref's own sigma_A via the acentric
`m = I1(X)/I0(X)`, `X = 2*alpha*|Fo||Fc|/(eps*beta)` gives mean m = 0.869 and
raised every site tested (ZN317 22.62 -> 23.50 at sr=3). Evidence is one
dataset, five sites -- treat as a hypothesis, not a settled win, and note it
would make the map *differ* from phenix rather than match it.

Peak z-scores are `peak_report`'s **blob maximum** (matches `peaks.csv`), *not*
interpolation at the atom position -- on a 0.65 A grid those differ by ~20%
(ZN317: 22.41 blob max vs 17.78 interpolated at the atom; the map maximum sits
0.37 A off the atom).

| map | grid | ZN317 | CA318 | CA319 | CA320 | CA321 |
|---|---|---|---|---|---|---|
| torchref as-run | 0.65 A | 22.41 | 7.15 | 4.12 | 6.13 | 6.44 |
| protocol, no FOM | 0.65 A | 22.62 | 6.70 | 4.36 | 6.48 | 6.98 |
| protocol + FOM | 0.65 A | 23.50 | 7.13 | 4.39 | 6.75 | 7.04 |
| **protocol + FOM, sample_rate=5** | **0.39 A** | **24.26** | **7.36** | **4.74** | **7.19** | **7.32** |
| phenix | 0.38 A | 23.64 | 7.10 | 4.60 | 7.00 | 7.01 |

**One** real worker-side deficit, not two: the FFT grid is under-sampled
(`SAMPLE_RATE = 3.0` gives 0.65 A here). FOM weighting is *not* a deficit
relative to phenix -- both programs are unweighted; see the correction above.

### The grid-size trap (verified 2026-08-15)

phenix's `refine_001.mtz` has 179,350 rows spanning d = 1.00-81.40 A, but ANOM
is populated **only to 1.8 A**: all 147,050 rows with `1.0 < d <= 1.80` are NaN.
gemmi's `transform_f_phi_to_map` substitutes 0 for MTZ missing-number flags, so
the map is clean (`isfinite().all()` True, mean ~1e-13, std 0.018) -- the NaNs
never reach `rs.find_peaks`. **But gemmi sizes the FFT grid from the MTZ's
overall `d_min` (1.0 A), not from where ANOM is actually non-NaN (1.8 A).** So
phenix's map is built on a 0.38 A grid and torchref's 1.8 A MTZ on 0.65 A.

Same coefficients, grid forced both ways:

| map / grid | | ZN317 | CA318 | CA319 | CA320 | CA321 |
|---|---|---|---|---|---|---|
| phenix @ its default | 0.38 A | 23.64 | 7.10 | 4.60 | 7.00 | 7.01 |
| **phenix forced to 144x144x240** | **0.65 A** | **22.65** | **6.70** | **4.42** | **6.53** | **6.83** |
| torchref+FOM @ 144x144x240 | 0.65 A | 23.50 | 7.13 | 4.39 | 6.75 | 7.04 |
| torchref+FOM @ 240x240x384 | 0.39 A | 24.26 | 7.36 | 4.74 | 7.19 | 7.32 |

phenix's own peaks drop ~1 sigma on the coarse grid. The finer grid is
**legitimate, not fake resolution**: zero-padding in reciprocal space is exact
sinc interpolation in real space, the map stays band-limited at 1.8 A, and
`peakz` is a *max over grid nodes* -- so undersampling can only lose height.
The Zn maximum sits 0.37 A off the atom, about half a voxel at 0.65 A.
Raising `SAMPLE_RATE` therefore improves torchref's own peak estimate, it is not
merely cosmetic matching.

**The apples-to-apples comparison is same formula, same grid: torchref without
FOM at 0.65 A = 22.62 on ZN317 vs phenix forced to 0.65 A = 22.65. Equivalent.**
The entire apparent peak deficit was grid sampling. The anomalous signal in
abismal's merged data was never the problem.

(The FOM-weighted rows beat phenix at 4 of 5 sites at 0.65 A and all 5 at
0.38 A, but that is a departure from phenix's formula, not parity with it --
see the correction above.)

### The peak metric itself is broken

Two independent defects, both affecting the benchmark's headline number:
1. **Peak z-scores are not comparable between programs unless the grid is
   matched.** phenix is flattered by an accident of how it pads its MTZ.
2. The ">= 5 sigma peak count" is unstable: `rsbooster`'s flood-fill drops peaks
   at `sigma_cutoff=5` that it reports at 6.75 sigma when called with
   `sigma_cutoff=4`.

Use peak z-score at known sites, on an explicitly fixed grid (`exact_size=`),
for both programs.

## 5. Validated protocol

End-to-end, and transferred to a second dataset:

| | cxidb_81_small | cxidb_81 (full) |
|---|---|---|
| as-run | 0.2049 / 0.2205 | 0.1960 / 0.2136 |
| **protocol** | **0.2039 / 0.2124** | **0.1963 / 0.2046** |
| Rfree gain | 0.0081 | 0.0090 |

Same optimum (adp ~ 0.2), same magnitude -- not a single-dataset artifact.

### Phase 0 -- BLOCKER: rigid body silently wipes component sigmas

`refine_rigid_body` **reconstructs the ADP targets** (verified: `id(adp_target
["simu"])` changes across the call), resetting every component sigma to its
default. `r.weighting` survives, because the group weights live on the
Refinement, not on the target.

This means **adding rigid body (Phase 1) silently breaks the existing
`search_aniso_sigma`**: the worker sets `simu_sigma_aniso` inside `build()`,
before any refinement, so every trial would run at the default 1.0, the ladder
would collapse to noise, and the worker would still print a confident
`adp aniso sigma fitted: X`. Fix before or with Phase 1: set component sigmas
**after** the rigid-body step, and assert the read-back.

This bit me during the analysis -- a first `simu_sigma` ladder came back with
four byte-identical rows, which I nearly reported as "simu_sigma does nothing
for isotropic ADPs." It was the wipe. Re-run after the fix, it is a real dial
(see the table below).

### Phase 1 -- the two big wins (worker-side only, no torchref changes)

1. **Enable rigid body.** `ref.refine_rigid_body(iterations_per_step=100)` then
   `ref.get_scales()`, before the macrocycle loop. 6 parameters against 58k
   reflections -- cannot overfit. torchref's default `--rigid-body-iter=30`
   under-converges by its own docstring (`cli/refine.py:187-193`).
2. **Sweep the `adp` group weight** over `[0.05, 0.1, 0.2, 0.5]`, reusing
   `search_aniso_sigma`'s structure (refit every epoch, one-standard-error
   selection). torchref itself has **no** weight optimisation to lean on --
   `--weights` (`cli/_common.py:406-427`) is a manual JSON override and nothing
   in the package ever searches it; the only trace is a "Pending the R_free
   weight scan" TODO at `base_refinement.py:54`. So this has to live in the
   worker.

   Ladder (cxidb_81_small ep100, **with** rigid body; phenix for reference):

   | dial | Rwork | Rfree | gap | B max | B sigma |
   |---|---|---|---|---|---|
   | group 0.02 (default) | 0.2009 | 0.2162 | 0.0154 | 91.8 | 8.97 |
   | group 0.05 | 0.2017 | 0.2147 | 0.0130 | 76.9 | 7.88 |
   | group 0.1 | 0.2031 | 0.2138 | 0.0107 | 66.4 | 6.93 |
   | **group 0.2** | 0.2054 | **0.2134** | **0.0079** | **56.9** | **5.91** |
   | group 0.5 | 0.2108 | 0.2150 | 0.0041 | 46.9 | 4.43 |
   | *phenix* | *0.1973* | *0.2053* | *0.0080* | *51.0* | *5.94* |

   **Use the group weight, not a component sigma.** Two reasons. (a) It is
   rebuild-safe -- see Phase 0. (b) It beats every component dial on the B
   distribution, which is what actually distinguishes the models:

   | dial (component x10, group left at 0.02) | Rwork | Rfree | gap | B max | B sigma |
   |---|---|---|---|---|---|
   | `adp/sigd` x10 | 0.2008 | 0.2160 | 0.0152 | 87.2 | 8.75 |
   | `adp/sigd` x0 (OFF) | 0.2009 | 0.2163 | 0.0154 | 92.3 | 9.00 |
   | `adp/simu` x10 | 0.2028 | 0.2136 | 0.0109 | 87.6 | 7.74 |
   | `adp/locality` x10 | 0.2040 | 0.2144 | 0.0104 | 65.6 | 6.39 |
   | `simu_sigma` 0.4 | 0.2046 | 0.2129 | 0.0083 | 85.4 | 7.18 |

   `simu_sigma=0.4` reaches a marginally *lower* Rfree (0.2129 vs 0.2134, ~1
   paired sd -- not significant) but leaves outlier B's at 85 A^2, because it
   tightens bonded-pair smoothness while `locality` and `sigd` stay loose. The
   group weight turns `simu` and `locality` together and lands on phenix's B
   distribution (sigma 5.91 vs 5.94) and phenix's work/free gap (0.0079 vs
   0.0080). Note `simu_sigma=0.63` reproduces `adp/simu` x10 almost exactly, as
   the `w/sigma^2` gradient equivalence predicts -- a useful consistency check.

   **`adp/sigd` is inert.** Turning it off entirely moves Rfree by 0.0001 and
   B sigma by 0.03, despite being the largest ADP term by raw magnitude (1688 vs
   simu's 6.8 at a flat-B start). It is a per-atom distribution prior, nearly
   constant with respect to the individual B assignments that matter. Candidate
   for removal, and a reason not to waste a sweep axis on it.

   **Selection rule.** Rfree spans only 0.2134-0.2162 across the whole ladder,
   so selection needs care -- but it is *better* resolved here than the hewl
   aniso case (5,739 free reflections vs ~650). Paired bootstrap, 4,000
   resamples on the same reflections, against the w=0.2 winner:

   | pair | dRfree | sd | \|d\|/sd | P(0.2 better) |
   |---|---|---|---|---|
   | 0.2 vs 0.05 | -0.0014 | 0.00049 | 2.80 | 99.75% |
   | 0.2 vs 0.1 | -0.0004 | 0.00028 | 1.55 | 94.35% |
   | 0.2 vs 0.5 | -0.0016 | 0.00043 | 3.76 | 99.98% |

   (sd of a *single* rung's Rfree is 0.00268 -- 5-10x the paired sd, which is
   why the comparison must be paired.) Here the raw argmin is trustworthy, and
   the existing one-standard-error rule (tolerance 0.001) picks the same rung:
   only 0.1 ties with 0.2, and it breaks the tie toward more restraint. Keep the
   one-SE rule anyway -- it costs nothing and hewl showed the argmin is not
   always safe.

   Cost ~4 x 32 s per epoch. The aniso precedent says refit every epoch rather
   than caching (a cached epoch-1 fit is fitted on unconverged data), so budget
   for that rather than trying to be clever.

### Phase 2 -- scaling (worth ~0.002)

3. `nbins = 10 -> 30`, and call `scaler.setup_binwise_solvent_scale()` after
   `get_scales()` (phenix-style per-bin k_mask; the exponential form cannot
   reproduce the flat-then-cliff shape). Measured together: -0.0015 Rwork,
   -0.0007 Rfree.
4. **Least-squares scale refit before reporting R and writing the MTZ.**
   Recovers 0.0025-0.0038 in both R's. Take this upstream to HatPdotS as a bug
   rather than patching around it -- the alpha/k inconsistency is torchref's.

### Phase 3 -- maps (fixes the peak metric)

5. **SPECULATIVE, DEPRIORITIZED.** Weight `ANOM` by m (and `FWT`/`DELFWT` by m
   and D) -- computable from the sigma_A estimate torchref already fits.
   Originally listed as closing a gap with phenix; that premise was **wrong**
   (phenix does not FOM-weight ANOM either -- see the correction in section 4),
   so this would make torchref *differ* from phenix, not match it. It did raise
   every peak tested (ZN317 22.62 -> 23.50), but on one dataset and five sites.
   Note `FWT`/`DELFWT` are a different matter: phenix's 2mFo-DFc and mFo-DFc
   genuinely *are* m/D-weighted (that path does reach `map_calculation_helper`),
   so weighting those is real parity work -- unlike ANOM. Not yet checked
   against phenix's `combine`/`fo_fc_scales` in `map_tools.py`.
6. **DONE 2026-08-15**: `SAMPLE_RATE` 3.0 -> 5.0 and a `--peak-dmin` flag in
   `_torchref_worker.py`. `find_anomalous_peaks(..., dmin=)` drops rows beyond
   dmin before the FFT -- the only way to pin the grid, since gemmi has no dmin
   argument and sizes from the largest Miller index *present as a row*, NaN
   included. On torchref's own MTZ the cut is a no-op (31697/31697 kept); it
   exists to guarantee the invariant and to make the phenix-side comparison
   honest. Both now log the resulting grid.

   Sample-rate ladder (cxidb_81_small ep100, torchref+FOM, ZN317 peakz):

   | sr | 3 | 4 | 5 | 6 | 7 | 8 |
   |---|---|---|---|---|---|---|
   | spacing (A) | 0.65 | 0.49 | 0.39 | 0.33 | 0.29 | 0.24 |
   | ZN317 | 23.50 | 23.08 | 24.26 | 24.29 | 24.14 | 24.73 |
   | t (s) | 0.10 | 0.20 | 0.39 | 0.61 | 0.90 | 1.43 |

   It does **not** converge: `peakz` is a max over grid nodes, a biased-low
   estimator of the continuous maximum, and the bias only shrinks while
   jittering +-0.3 with node placement (sr=4 reads *below* sr=3). Map sigma is
   constant (3e-5) at every rate, so the z-score denominator is unaffected --
   this is purely a better-sampled numerator. 5 buys ~0.5 sigma for 0.3 s/epoch
   and is where we stop; because it is not converged it must stay FIXED and
   identical on both sides of any comparison. Removing the bias rather than
   shrinking it needs subpixel refinement (quadratic fit around the max node)
   in rsbooster.

   Side effect worth knowing: at sr=5 the **as-run** epoch-100 map finds
   **5 peaks** >= 5 sigma (ZN317, CA318, CA321, CA320, HOH1002) against 3 at
   sr=3 -- matching phenix's count. The "3 vs 5 peaks" deficit was largely a
   sampling artifact, not map quality.

7. Replace the ">= 5 sigma peak count" benchmark metric with peak z-score at
   known sites, on a common grid pinned by dmin on **both** sides. Verified:
   cutting phenix's `refine_001.mtz` to d >= 1.8 yields grid (144,144,240) at
   sr=3, byte-identical to torchref's, and reproduces the forced-size numbers.
   Fully controlled at sr=5, both on (240,240,384):

   | | ZN317 | CA318 | CA319 | CA320 | CA321 |
   |---|---|---|---|---|---|
   | phenix, d>=1.8 | 23.38 | 6.87 | 4.61 | 6.96 | 6.92 |
   | torchref+FOM | **24.26** | **7.36** | **4.74** | **7.19** | **7.32** |

   The legacy phenix path (`callbacks/peak_finder.py`) shells out to the
   `rs.find_peaks` CLI and has **not** been given the same treatment -- it is
   still confounded. Fix it before quoting any phenix-vs-torchref peak number.

## 5b. torchref.refine CLI, out of the box (2026-08-15)

Ran the stock CLI on the same epoch-100 data + 2tli, `--wavelength 1.5418`,
~28 s on GPU. The CLI has **no ADP-only mode** (`--mode separate|everything`,
both refine xyz) and **no flat-B reset**, so it cannot reproduce the worker's or
phenix's protocol.

torchref's own reported R:

| protocol | Rwork | Rfree | gap |
|---|---|---|---|
| CLI default | 0.1718 | 0.2166 | 0.0448 |
| CLI `--with-rigid-body --rigid-body-iter 100` | 0.1717 | 0.2169 | 0.0452 |
| CLI `--weights '{"adp":0.2}'` | 0.1754 | 0.2150 | 0.0396 |
| CLI `--weights '{"adp":0.2,"geometry":1.0}'` | 0.1828 | 0.2143 | 0.0315 |
| CLI rigid body + those weights | 0.1830 | 0.2142 | 0.0312 |
| **worker protocol (ADP-only + rb + adp 0.2)** | 0.2054 | **0.2134** | 0.0079 |
| phenix | 0.1973 | 0.2053 | 0.0080 |

Same models on the common scaler (`phenix.model_vs_data`):

| model | Rwork | Rfree | gap |
|---|---|---|---|
| CLI default | **0.1678** | 0.2113 | 0.0435 |
| CLI adp 0.2 | 0.1710 | 0.2087 | 0.0377 |
| CLI adp 0.2 + geom 1.0 | 0.1782 | 0.2082 | 0.0300 |
| **worker protocol** | 0.1976 | **0.2063** | 0.0087 |
| phenix | 0.1973 | 0.2053 | 0.0080 |

Model characteristics:

| model | xyz rmsd | max dxyz | B sd | B max | `<\|Bi-Bj\|>` | bond rmsd | angle rmsd | clash |
|---|---|---|---|---|---|---|---|---|
| CLI default | 0.201 | 1.235 | 7.90 | 84.7 | 3.73 | 0.013 | 2.045 | 2.72 |
| CLI rb+tight | 0.184 | 0.955 | 5.42 | 57.6 | 1.80 | 0.008 | 1.449 | 1.68 |
| worker protocol | 0.063 | 0.089 | 5.91 | 56.8 | 2.00 | 0.015 | 2.542 | 2.72 |
| phenix | 0.060 | 0.094 | 5.94 | 51.0 | 2.18 | 0.015 | 2.542 | 2.72 |

**Conclusions.**

1. The CLI wins Rwork decisively (0.168 vs phenix's 0.197 on a common scaler)
   and *loses* Rfree (0.211 vs 0.205). Work/free gap 0.044 against 0.008.
2. **It is overfitting, not model damage.** Geometry is fine or better than the
   deposited model -- bond rmsd 0.013 (0.008 at geometry 1.0) vs 0.015,
   clashscore 1.68-2.72. The CLI moves atoms 0.20 A rmsd, max 1.24 A: real,
   geometrically valid coordinate refinement that fits noise in the amplitudes.
3. **Tightening weights helps but does not close it.** Best CLI Rfree is 0.2082
   on the common scaler, still behind the ADP-only worker protocol (0.2063) and
   phenix (0.2053) -- while refining ~4x the parameters.
4. **`--with-rigid-body` is a no-op once xyz is refined** (0.2166 -> 0.2169).
   Rigid body only earns its keep in the ADP-only protocol, where it is worth
   0.005 Rfree. Do not read the CLI result as evidence against Phase 1.
5. Not like-for-like, in two ways that both flatter the CLI: phenix was
   configured for `rigid_body + individual_adp` only, and the CLI inherits
   2tli's already-refined B-factors (mean 26.85) instead of the flat B=20 reset
   phenix used.

This is direct confirmation of the worker's design choice to skip `refine_xyz()`
-- previously supported only by the hewl measurement in the code comment
(0.1576/0.1761 with xyz vs 0.1671/0.1628 without).

## 6. Found but not acted on

- `lbfgs_refinement.py:285` calls `solvent.update_solvent()` without clearing
  `Scaler._f_sol_raw`, so the per-cycle mask rebuild is **silently discarded**
  (`ScalerBase.forward` only recomputes when the cache is None; `solvent.py:416-420`
  documents the requirement). Inert today because coordinates never move, but
  **becomes live once rigid body is enabled** -- Phase 1 must clear the cache or
  call `get_scales()` after the rigid-body step.
- `refined.mtz` writes R-free flags with **1 = work, 0 = free**, the opposite
  polarity from the input MTZ. Anything re-refining from it with inferred
  polarity will swap work and free.
- If no FreeR column is present torchref **generates** one at `free_fraction=0.02`
  (`reflection_data.py:1016-1022`) -- unusually small vs the conventional 5%.
  The benchmark passes `--r-free-mtz` so this does not bite, but it is a landmine.
- `screen_solvent_params()` (phenix-like 15x15 k_sol/B_sol grid,
  `scaler_base.py:430-564`) exists but is **never called** on the single-dataset
  path. Tested: no help here.
- Freeing the `adp/scaler_log_scale` and `adp/scaler_U` penalties: **no effect**
  (0.2050 / 0.2205, identical to baseline). The hypothesis that they force the
  overall falloff into the atomic B's was wrong.
- torchref's sigma_A floors at 0.910 where phenix's alpha drops to 0.80 in the
  outer shell -- its model-error estimate is over-optimistic at high resolution.
  Plausible cause of the residual scaler gap; not chased down.
- The TorchRef fork at `/home/kmdalton/opt/TorchRef` is **exactly in sync** with
  `HatPdotS/TorchRef` (0 commits either way at `0053149`). The older
  "fork goes stale" note is out of date.
- `example_notebooks/targets_and_weighting.ipynb` documents a `ComponentWeighting`
  scheme and a `bhattacharyya` default that **no longer exist**. Do not trust it.

## 7. Reproducing

Ablation driver, LS-rescale probe and FOM experiment were written to a
job-scoped scratchpad and are not preserved. To rebuild: construct
`LBFGSRefinement` with the worker's kwargs (`adp_mode="isotropic"`,
`target_mode="ml"`, `scale_target="ml_noalpha"`, `french_wilson=False`),
flat-reset B via `PositiveMixedTensor` + `model.reset_cache()`, override
weights with `ManualWeighting(dict(DEFAULT_GROUP_WEIGHTS, **{"adp": w}))`, and
score with `phenix.model_vs_data <pdb> input_data.mtz
f_obs_label="F(+),SIGF(+),F(-),SIGF(-)" r_free_flags_label=R-free-flags`.
One refinement is ~32 s on CPU.
