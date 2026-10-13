# abismal 2.0 (PyTorch) — handoff

Written 2026-10-13 for the agent that continues this work, expected to run on NERSC Perlmutter.
It covers the TensorFlow → PyTorch port and the benchmark experiments since then. Read the
**Working agreements** section before doing anything.

## 1. State at a glance

- The port is **done**. Branch `torch` = `origin/torch` = `b2153d5`. TensorFlow is gone from the code,
  packaging, CI and docs. The full test suite passes: 686 passed, 7 skipped, 4 xfailed with `--ignore=tests/gui`.
- Defaults on `torch` are unchanged by the experiments: width (`--d-model`) 256, depth 5, relu/relu,
  Student t ν = 32, `--metadata-noise-factor 0.1`, lr 1e-3, batch 100 images, 1000 steps/epoch, 32 MC samples.
  On CUDA, `torch.compile` and bfloat16 are on by default.
- The user leans towards **l2 pre-activation + swish activation as the default "when time isn't an issue"**.
  Nothing has been changed on `torch` yet; that is the user's call.
- The newest results are `results/posterior_features/` (section 6.15). The first Perlmutter job is the `deep-context` test (section 7).

## 2. Repositories and branches

### abismal (`ssh://git@github.com/rs-station/abismal`)

| branch | commit | on origin | what |
|---|---|---|---|
| `torch` | b2153d5 | yes | the port; the base for everything below |
| `fix-rice-seed` | 2d32398 | no (merged into torch) | Rice sampler drew both normals from one seed |
| `split-half` | d62ace2 | yes | fit half datasets by default and write them every epoch (2 ahead of torch) |
| `posterior-features` | 55db98e | yes | `--posterior-features` (section 6.15) |
| `deep-context` | d09f8d8 | yes | `posterior-features` + `--deep-context` (section 7, first test) |
| `frozen-scale-probe` | e9d9592 | yes | `--scale-init-file` also loads the standardization layers |
| `noise-scale-only` | 69cd5eb | yes | metadata noise added to the scale branch only |
| `background-model` | 53f5f5a | yes | background latent B, env flags `ABISMAL_BACKGROUND`, `ABISMAL_BG_*`, `ABISMAL_POOL_ONLY` (shelved) |
| `error-model` | 42bfaa5 | yes | Ev11 (joint) and MM24 (alternating) error models (shelved) |
| `fixed-batches` | 705aecb | yes | fixed-shape batches by sampling reflections per image |

The experiment branches are throwaway; they stay off `torch` deliberately. All were pushed to origin on 2026-10-13 so they can be used on Perlmutter. This file lives on the `handoff` branch.

Commits on `torch` since `gui`, oldest first:

```
2d32398 Draw the Rice sampler's two normal components from independent seeds
9751e55 Make abismal.io framework free, keeping the streaming pipeline
95fcb71 Reimplement abismal.distributions in PyTorch
641da1c Port the model, training loop and CLI from TensorFlow to PyTorch
d1cdc10 Test the training loop and per-image operator selection; add a step benchmark
6f7d6fc Drop TensorFlow from packaging, CI, install scripts and docs
569c2b8 Give the test model builders their own module
db0ccd0 Add abismal.convert, which turns abismal 1.x .keras models into .pt files
f011fd3 Use expandable segments for the CUDA caching allocator
0665dd5 Make torch.compile work on the whole step, and add bfloat16 networks
b21e772 Compile and use bfloat16 by default on CUDA
adfbf75 Weight per-epoch metrics by the number of images in each batch
a3daac5 Hide the GPU from torchref workers that refine on the CPU
ad4ea04 Set the allocator through torch's current API, falling back to the old one
b2153d5 Weight rows of a batch for context and loss; optionally validate on held-out reflections
```

### abismal-benchmarks (`ssh://git@github.com/rs-station/abismal-benchmarks`)

- Checked out on `dev` (HEAD cc35bef). It has **uncommitted edits that the agent did not make** to
  `benchmarks/config/{cxidb_81_small,hewl}.sh`, `benchmarks/job.sh` and `scripts/plot_progress.py`. Leave them alone.
- Any change to this repo goes on a **new branch**.
- `results/` is **gitignored**. Every experiment script, analysis and summary lives only on the
  workstation, at `/media/kmdalton/3bd97678-a8fe-47cb-bdd2-b2fdcc5569b1/data/xtal/abismal-benchmarks/results/`.
  `make_handoff_bundle.sh` packs the portable part (on branch `perlmutter-handoff` of the benchmarks repo); see section 9.

### TorchRef (`https://github.com/HatPdotS/TorchRef`, 0.6.3, commit 0053149 on its `main`; the kmdalton fork's `main` is behind)

Used for refinement through `abismal/callbacks/_torchref_worker.py`, always run on the CPU in a
separate process.

## 3. The port

The plan is in `~/.claude/plans/i-am-finding-it-snug-cook.md` on the workstation. In short:

- **Data pipeline** (`abismal/io`): pure numpy, no framework import. A producer thread drains the ray
  generator. Images go to train or test by a splitmix64 hash of (index, seed). The cache grows while
  epoch 1 trains, then replays. A reservoir shuffle reproduces tf.data `shuffle` semantics.
  - Batches are ragged: flat arrays plus `image_id`, built from whole images.
  - The train/test split is deterministic, but not bit-identical to TF.
- **Distributions** (`abismal/distributions`): FoldedNormal, Rice, Nakagami, TruncatedNormal and
  RiceWoolfson are written in torch. They use implicit reparameterization by the detach trick.
- **Model:** `ImageScaler` uses leave-one-out pooling via `index_add_`. `VariationalMergingModel`
  selects the reindexing operator per image. A custom `AbismalAdam` does lazy, masked updates of the
  posterior.
- **Training:** a hand-written loop and callbacks, `.pt` checkpoints, and `abismal.convert` for
  1.x `.keras` files. CLI flag aliases are kept for the GUI.
- **Performance:** whole-step `torch.compile`, bfloat16 networks with the pooling done in float32,
  and expandable CUDA allocator segments.
- **Parity with TF** (`results/torch_port/compare.md`, 100 epochs, no metadata noise):

  | dataset | | Rfree | max peak z | CC½ |
  |---|---|---|---|---|
  | HEWL | TF | 0.1544 | 19.31 | 0.9996 |
  | HEWL | torch | 0.1549 | 20.11 | 0.9996 |
  | cxidb_81_small | TF | 0.2258 | 19.69 | 0.9869 |
  | cxidb_81_small | torch | 0.2223 | 21.34 | 0.9885 |

  cxidb_61, cxidb_62 and full cxidb_81 are also in that file.

## 4. Benchmark harness (how every experiment was run)

- **Datasets:**
  - `cxidb_81_small`: SFX stills, one DIALS chunk `r0011_t016_rg013_chunk000_reintegrated.{expt,refl}`.
    About 2860 training images (10% of images are held out for validation) and 342 reflections per image. Anomalous Zn and Ca. Model 2tli, λ 1.5418 Å.
  - `hewl`: rotation, `unmerged.mtz`, about 1300 training frames and 286 reflections per frame. Sulfur
    anomalous. Model RTSAD_HEWL_refine_25, λ 1.892 Å.
- **Command lines:** `results/torch_port/build_args.py LOG OUTDIR` rebuilds an abismal command line
  from a run's `abismal.log`. The base logs are `results/adam_nonoise/{cxidb_81_small,hewl}/abismal.log`.
  - Each `run.sh` strips the options it sets itself (`--torchref*`, `--r-free-mtz`, `--studentt-dof`,
    `--metadata-noise-factor`, `--seed`, `--epochs`, ...) with a small inline Python filter, then adds
    the arm's options.
  - Runs launch as `python -c 'from abismal.command_line.abismal import main; main(sys.argv[1:])' ARGS`
    with `PYTHONPATH` set to a **pinned worktree**. The pip-installed abismal in the env points
    elsewhere, so never rely on it.
- **Refinement** happens after training, from the saved `asu_0_epoch_N.mtz`, three at a time:
  ```
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python $WORKTREE/abismal/callbacks/_torchref_worker.py \
    --mtz RUN/asu_0_epoch_N.mtz --pdb MODEL --wavelength W --out-dir RUN/torchref_0_asu_0_epoch_N --device cpu \
    --macro-cycles 5 --z-score-cutoff 5.0 --r-free-mtz DATA/reference_data/r-free-flags.mtz \
    --adp-mode auto --adp-aniso-sigma auto --rigid-body-iter 30
  ```
  The worker writes `peaks.csv` and `stdout.txt`, the latter with `Rwork=`/`Rfree=` lines.
  `--allow-overlap` is a callback option, not a worker flag.
- **Metrics:**
  - cxidb: **Zn** is the maximum ZN peak z, **Ca** the mean CA peak z, plus Rfree at a given epoch.
  - HEWL: **S** is the mean CYS/MET peak z; "best" means best over epochs refined every 5, plus Rfree.
  - Training: `history.csv` has CCpred, val_CCpred, NLL, val_NLL, Σ_mean, |∇s| and KL_Σ.
  - **val_CCpred does not track map quality. Prefer NLL / val_NLL** (more dynamic range) and the refined maps.
- **Seed noise:** cxidb_81 seeds differ by about 0.4 in Zn and about 0.0005 in Rfree. Almost
  everything since the port is **single-seed** (seed 1).
- **Timing on one RTX A6000:**
  - 20 cxidb epochs alone take about 7.5 min.
  - Five width-256 runs sharing the GPU take about 2 h for 100 epochs.
  - Width 1024 costs about 6× the default per epoch.
  - Refinement takes 2–3 min per map on 4 CPU threads.

## 5. Working agreements (the user's standing instructions)

- Progress updates go to the terminal only. Be concise and lead with the result.
- **Push only when asked.** Changes to the benchmarks repo go on a new branch.
- **Avoid creeping featuritis.** Experimental knobs stay off `torch`, in throwaway worktrees and branches.
- Subagents work in worktrees. Don't spawn agents unless asked.
- Ask if an instruction is unclear. Don't start new experiments without the user's direction, but
  queuing the obvious run the user asked about is fine.
- The user is a crystallographer and abismal's author. Use crystallographic terms freely, and give a
  recommendation rather than a survey.

## 6. Experiments and conclusions

All are in `results/<name>/` with a `run.sh`, usually an `analyze.py`, and a `summary.md`.
"Reference" usually means `studentt_dof_cxidb81/dof_32` (cxidb, canonical settings) or
`posterior_lr/lr_x1` (HEWL, canonical settings).

1. **torch_port:** parity with TF; see section 3.
2. **metadata_noise** (0 vs 0.1, two seeds, cxidb): noise 0.1 gives about +2 Zn (21.1 → 23.3) and
   Rfree 0.2236 → 0.2157. This is why 0.1 is the default.
3. **rice_posterior:** Rice vs FoldedNormal on cxidb are equivalent within seed noise. The Rice seed
   bug made no measurable difference.
4. **posterior_lr / posterior_beta2** (HEWL): a higher posterior learning rate or a lower β₂ does not
   help. The defaults (×1, β₂ = 0.999) are best or tied.
5. **error_model** (HEWL): Ev11 joint error-model refinement is ill-posed (S collapses to about 11–13).
   MM24 alternating calibration works but doesn't beat the reference. Shelved.
6. **holdout_reflections / split_halves / split_halves_hewl:** reflection-level validation and
   half-dataset statistics.
   - Train NLL keeps falling while val NLL rises from about epoch 10, i.e. overfitting of scales.
   - On HEWL, CC½ saturates (about 1.000) and CCanom rises while S drifts down. That suggests
     half-dataset CCs are a poor stopping criterion here.
7. **normal_collapse** (HEWL, `--studentt-dof inf`):
   - With the normal likelihood, Σ → 0 in epoch 1.
   - The `exp_*`, `init0.01_*`, `init0.1_*`, `out0_*` and `lognormal_kl0_*` arms in `summary.md` all still collapse.
   - **l2 or layer pre-activation normalization, or a lognormal scale posterior with KL, does prevent it.**
8. **background_model / background_normal / background_pooled:** a per-observation background
   latent B with KL to N(0, τ²) (τ from DIALS background variance).
   - **Null under t** (posterior ≈ prior, |B|/σ ≈ 0.07). It only rescues the normal-likelihood collapse.
   - Gradient probes showed the rescue comes from the background head's gradient into the shared
     trunk, not from the additive offset.
   - Moving the background columns to the pooling branch only changed nothing.
   - **The user dropped this line of work.** A report artifact exists:
     https://claude.ai/artifact/Kwt1sFJvTXk7wjVgeaEdzo
9. **error residuals** (`results/error_model/residuals.py`):
   - Normalized residuals are symmetric with heavy tails: a fitted t has ν ≈ 4.5 on HEWL and 2.2 on cxidb.
   - Test images have residuals about 2× wider (the scale model overfits training images).
   - On cxidb, weak reflections are right-skewed and the strongest are over-predicted.
10. **studentt_dof** (HEWL) and **studentt_dof_cxidb81:** ν from 1 to 1024.
    - Any ν from 16 to 512 is fine on both datasets.
    - On HEWL, S peaks at ν = 8 and Rfree flattens from ν = 64.
    - On cxidb, both improve up to ν ≈ 128–512 (Zn 23.5 at ν = 32, about 25 at ν ≥ 256; Rfree 0.2148 → 0.2122).
    - The residual-fitted ν is a bad choice. The user considers 32 or 64 a fine default.
11. **metadata_noise_sweep** (σ in 0, 0.01, 0.03, 0.1, 0.3, 1.0, both datasets):
    - 0.1 is at or near the optimum on both.
    - 0.3 converges much faster on cxidb (Zn 23.0 by epoch 20) with Rfree 0.2124, but not on HEWL.
    - 1.0 is bad.
    - **noise_scale_only:** restricting the noise to the scale branch makes no difference.
    - **spot_spacing:** σ = 0.1 ≈ median within-image nearest-neighbour distance / √(2k) in
      standardized metadata space (0.115 cxidb, 0.102 HEWL). This supports the user's hypothesis that
      noise stops the scale network memorizing spots. The user dislikes the arbitrariness of the noise
      scale and may revisit a data-derived σ.
12. **width_sweep** (relu, widths 16–1024):
    - Wider networks start faster but plateau lower (overfit). Width 128 matches 256.
    - **Width 1024 collapses to Σ = 0 in epoch 1 on cxidb even under t₃₂.** HEWL at 1024 is fine.
13. **width1024_probe:** l2 pre-activation rescues width 1024; swish alone does not (it collapses in epoch 2–3).
    - l2 + relu reaches Zn 23.7 by epoch 40, versus about 90 epochs for the default.
    - l2 + swish is slow but ends at Zn 24.1 and Rfree 0.2135, the best ν = 32 result.
    - **width_l2_sweep** (widths 32–512 × {l2 + relu, l2 + swish}, both datasets): L2 does nothing at
      ≤ 128, helps slightly at 256 (near seed noise), and removes the overfitting stall from 512 up.
      l2 + swish is slower at every width and wins only when wide. On HEWL it gives the best Rfree but
      lower S at 50 epochs, possibly undertrained.
14. **convergence_probe:**
    - Freezing the scale network at its converged state (`frozen-scale-probe` branch) lets the
      structure factors converge in **about 20 epochs, versus 80–90 from scratch**. Zn reaches 24.4 at
      epoch 20, then degrades slowly. **Learning the scale network is the rate limiter.**
    - `--init-scale` 0.01–1.0 has no effect; the posterior width is set by the data within about 10 epochs.
    - Independent converged runs agree to an anomalous-difference CC of about 0.92.
15. **posterior_features** (in progress at handoff, `posterior-features` branch):
    - `--posterior-features` appends each reflection's current posterior E[F] and SD[F] (no gradient,
      identity operator, zero for invalid reflections, no rescaling) to the **pooling-branch** input only.
    - Arms: relu and l2 + swish at width 256, on both datasets. Baselines are `dof_32`, `lr_x1` and
      `width_l2_sweep/*_l2swish_256`.
    - **Result (finished 2026-10-13):** val NLL is lower at the end in all four runs (cxidb relu 12.50
      vs 13.11, cxidb l2 + swish 10.66 vs 11.80, HEWL relu 2.34 vs 2.69, HEWL l2 + swish 1.68 vs 2.06).
    - **Maps:**
      - l2 + swish improves on both datasets: cxidb Zn 23.70 vs 23.35, ahead at every epoch from 5;
        HEWL S 18.07 vs 17.74, ahead from epoch 30. Rfree is unchanged.
      - relu only speeds up the first ~10 epochs (cxidb Zn 19.7 vs 18.8 at epoch 5), then ends level
        or slightly worse (cxidb Zn 23.24 vs 23.51, Rfree 0.2158 vs 0.2148).
    - Single seed; the differences are near seed noise, but the l2 + swish sign is consistent over many epochs.

Report artifact covering items 8 and 10–14 (built by `results/reports/2026-10-12/build.py` from
`data.py` → `data.json` + `template.html`): https://claude.ai/artifact/24Uern4xY7nUwkq7TWsx6y

## 7. Open threads, roughly in priority order

**Deep context: done on the workstation on 2026-10-13 (`results/deep_context/`, plots `maps.png` and `nll.png`), so it no longer needs to run first on Perlmutter.**
- Two seeds per arm, both datasets: l2 + swish + posterior features, width 256, with and without `--deep-context`.
- **The maps improve with deep context:**
  - It converges faster. HEWL reaches the baseline's epoch-40 S and Rfree by about epoch 20. cxidb Zn is
    +0.5 to +1.5 over epochs 10–30, and Rfree is lower at nearly every epoch on both seeds.
  - It ends equal or better: cxidb Zn 23.80 / 24.15 vs 23.70 / 23.54, Rfree 0.2138 / 0.2140 vs
    0.2143 / 0.2148. HEWL ends level.
- **Image-level validation NLL disagreed.** Deep context's validation NLL rises after about epoch 20
  while train NLL keeps falling, yet the maps keep improving. Don't use image-level val NLL to judge
  architecture changes.
- The cost is about +15–17% time per epoch and +1% GPU memory.
- Original plan, kept for reference:
- Branch `deep-context` (d09f8d8) adds `--deep-context`:
  - The scaler's two tracks run in step. After every image-track layer, its leave-one-out pool is
    added, through a per-layer linear projection, to the scale track ahead of the layer of the same depth.
  - By default the image track is pooled once, at the end, into the scale track's input.
  - With the CLI default `share_weights=True` the two tracks share their feed-forward layers.
  - `tests/merging/test_deep_context.py` checks that a reflection's own Iobs, SigIobs and posterior
    features never reach its own scale. A mutation check confirmed it catches a leaky pool.
- Run it on both datasets, width 256, `--pre-activation l2 --activation swish --posterior-features`,
  with and without `--deep-context`, **seeds 1 and 2**. That's 8 runs, i.e. 2 GPU nodes at 4 runs each.
- Settings otherwise canonical: ν 32, noise 0.1, lr 1e-3; cxidb 100 epochs, HEWL 100 epochs, so that
  l2 + swish isn't undertrained.
- Re-run the without-deep-context arms on Perlmutter rather than reusing the workstation baselines:
  the A100 and bfloat16 numerics differ.
- Refine cxidb at epochs 2, 5, 10, 20, ..., 100 and HEWL every 5 or 10.
- Compare Zn/Ca/S, Rfree and train/val NLL by epoch, and the time per epoch.
- The user also asked about a **bidirectional** version, where the pooled context feeds back into the
  per-reflection track. Don't implement it before the user decides. The design discussed:
  - **Why it leaks:** after round 1 every other reflection's state contains *r*'s data through its own
    leave-one-out context, so round 2's pool returns *r*'s data to *r*.
  - Exact per-reflection exclusion costs O(n²). Keeping all cross-reflection paths linear is exact but
    no more expressive than one-way.
  - **Proposed: cross-fitting with random folds.**
    - Each step, split each image's reflections at random into K folds (K = 2).
    - The context for fold *a* comes from bidirectional message passing over the other folds only, so
      no amount of hops can return *r*'s data.
    - Implement by tiling the batch K times with virtual image ids (image, excluded fold), pooling
      weight 0 for the excluded fold, and `num_images × K` pool segments.
    - At inference, average over a few fold draws.
    - The same leave-one-out test applies.
  - The motivation is learned iterative scaling: rough scale, then residuals against Σ·E[F]² using
    the posterior features, then a refined scale.
  - The user noted that reflection subsampling, which this resembles, never helped overfitting in the past.


1. `posterior_features` is analysed (section 6.15). It helps l2 + swish, so test it at width 512–1024 and with a second seed.
2. Decide the defaults with the user: l2 pre-activation (cheap) and/or l2 + swish. Before changing
   any default, run a **second seed** at width 256, relu vs l2 + relu, on both datasets, plus a longer
   HEWL run (≥ 100 epochs) for l2 + swish.
3. Speeding up the scale network, the identified bottleneck:
   - a separate, higher learning rate for the scale network;
   - warm starts;
   - width-1024 relu with lr scaled by 256/width, to separate width from step size in the collapse.
4. A data-derived metadata-noise scale, tested on a dataset with a very different spot density.
5. Use Perlmutter for **multi-seed** versions of the close calls, and for the larger datasets
   (full cxidb_81, cxidb_61, cxidb_62), which were too slow locally.
6. Housekeeping the user hasn't decided:
   - cleaning up old pinned worktrees on the workstation;
   - the `--studentt-dof inf` warning;
   - a TorchRefRunner thread cap.

## 8. Gotchas learned the hard way

- `pkill -f PATTERN` or `kill $(pgrep -f PATTERN)` matches your own shell, whose command line contains
  the pattern, and kills it (exit 144). Identify processes by PID and `/proc/PID/cmdline` first.
- `tail -F` on a glob only watches files that already exist. Poll instead. Foreground `sleep` is
  blocked in Claude Code; use `run_in_background` until-loops.
- `--scale-init-file` loads only the scale network. A frozen network then sees inputs standardized
  with re-estimated statistics. `frozen-scale-probe` fixes this.
- Metadata noise is applied before the scale/pooling split; that was harmless (item 11).
- `batch_normalize` is off in the benchmark settings (`--batch-normalize` is store_true).
- `ImageScaler.forward` returns a tuple on `background-model`; `results/error_model/residuals.py`
  assumes a tensor.
- A run that dies by collapse has Σ_mean ≈ 1e-12 and |∇s| = 0 from then on. It never recovers, so
  kill it to free the GPU.
- The history CSV also logs GPU memory columns; metric columns follow them.

## 9. Setup on Perlmutter (the agent does this)

The user copies the bundle (`abismal_handoff_YYYY-MM-DD.tar.gz`, built by `make_handoff_bundle.sh` on
branch `perlmutter-handoff` of abismal-benchmarks) to Perlmutter and starts you in a clone of abismal
on the `handoff` branch. Everything else is yours. Ask the user only for what you can't find out:
- where the bundle is, if it isn't obvious (`ls $SCRATCH /global/cfs/cdirs/*/ 2>/dev/null`);
- the NERSC account to charge (`-A`). `sacctmgr show assoc user=$USER format=account%20` may list it;
- whether results should live on `$SCRATCH` (fast, purged) or CFS (persistent).

Check module names, QOS names, GPU constraints and CUDA versions against NERSC's docs and the live
system (`module avail`, `nvidia-smi` on a GPU node). The notes below are from memory.

1. **Unpack the bundle** into `$SCRATCH/abismal-benchmarks`. It contains `data/` and `results/`,
   whose paths match those used throughout this file. Copy `results/HANDOFF.md` aside; it is this file.
2. **Get the code:**
   - abismal: this clone. `git fetch` to get all branches. The experiment branches are on origin.
   - TorchRef: `git clone https://github.com/HatPdotS/TorchRef` next to abismal and check out 0053149
     (or newer if the user agrees).
   - abismal-benchmarks itself isn't needed to run experiments. The bundle has everything under `results/`.
3. **Environment:** use a conda env (`module load conda`, or micromamba) with Python 3.12.
   - Install a CUDA-12 torch wheel that matches the GPU nodes' driver, then `pip install -e <abismal> -e <TorchRef> pytest`.
   - Run `python -m pytest -q tests --ignore=tests/gui` on the login node and expect 686 passed.
   - The workstation's package versions are listed at the end of this file, for reference.
4. **Smoke test on a GPU:** get a short interactive allocation (e.g. `salloc -A <acct> -C gpu -q interactive -N 1 --gpus 1 -t 30`).
   - Train 2 epochs on `tests/data/stills.{expt,refl}` (`-d 2.0 --epochs 2 --steps-per-epoch 20`).
   - Run one torchref refinement on the output MTZ, with the worker command in section 4.
5. **Repoint the experiment scripts.** Every `results/*/run.sh` hard-codes three things:
   - `B=/media/kmdalton/...` → the new benchmarks root;
   - `PY=` → the new env's python;
   - `PYTHONPATH=` → pinned worktrees (`~/opt/sandbox/abismal-*`). Recreate the pins you need with
     `git worktree add ../abismal-<name> <commit>`. The commits are in each script's header comment.
   - Do this with a script, not by hand. Keep the originals.
6. **Write a Slurm launcher** for the harness, saved under `results/` with a short README:
   - One GPU node has 4 A100s and 64 CPU cores. Run four training arms per node, one per GPU with
     `CUDA_VISIBLE_DEVICES=i`.
   - Then refine on the node's CPUs (`CUDA_VISIBLE_DEVICES=`, 4 threads per worker, about 8 workers).
   - Use `-C gpu -q regular` for multi-hour sweeps and the shared QOS for single-GPU jobs.
   - Monitor with `squeue`/`sacct` from the login node. Never train on the login node.
7. **Then start the work:** item 1 of section 7, analysing `results/posterior_features`. If the bundle
   was built before those refinements finished, their results are missing (it doesn't carry MTZs).
   Ask the user for a fresh bundle, or re-run the four arms on Perlmutter (`results/posterior_features/run.sh`,
   code on the `posterior-features` branch).

Report setup progress to the user in a few lines at each milestone.

### Reference: workstation environment

The workstation env had torch 2.13.0+cu130, ray 2.57.0, reciprocalspaceship 1.0.7, gemmi 0.7.3,
numpy 2.3.5 and TorchRef 0.6.3. It had one RTX A6000 (48 GB).
