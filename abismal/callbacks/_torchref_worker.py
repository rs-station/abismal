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
"""
import argparse
import sys
from pathlib import Path

# Invoked by file path, Python prepends this script's directory
# (abismal/callbacks/) to sys.path, where the sibling ``torchref.py`` callback
# module would shadow the installed ``torchref`` package. Drop that entry so
# ``from torchref import ...`` resolves to the real package.
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _here)]

# Flat starting B-factor (A^2) every atom is reset to before ADP refinement.
RESET_B = 20.0

# Number of scaling + ADP-refinement macrocycles.
MACRO_CYCLES = 5


def _is_anomalous(ds):
    """Detect anomalous data from the presence of Friedel-split columns."""
    return "F(+)" in ds.columns


def prepare_data(mtz_path, pdb_path, out_dir):
    """Read the abismal MTZ, prepare it for torchref, and write a temp copy.

    For anomalous data the Friedel mates are stacked into a single ``F``/``SIGF``
    column set with signed HKLs (TorchRef tracks the sign per row for the
    structure-factor calculation) and unobserved mates that stack as NaN are
    dropped. The data cell is stamped to the model cell so refinement, scaling,
    the output PDB and the map MTZ all share one cell.

    Returns ``(input_mtz_path, anomalous)``.
    """
    import gemmi
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(mtz_path))
    anomalous = _is_anomalous(ds)
    if anomalous:
        ds = ds.stack_anomalous()
        if "F" in ds.columns:
            n_before = len(ds)
            ds = ds.loc[ds["F"].notna()]
            print(
                f"dropped {n_before - len(ds)} unobserved Bijvoet mates -> "
                f"{len(ds)} reflections",
                flush=True,
            )

    # Drop intensities (and their sigmas) so torchref refines directly against
    # abismal's already-merged amplitudes. Otherwise torchref auto-detects the
    # I columns and re-derives F from them via French-Wilson, ignoring F/SIGF.
    intensity_cols = [
        c for c in ds.columns if isinstance(ds[c].dtype, rs.IntensityDtype)
    ]
    sigi_cols = [f"SIG{c}" for c in intensity_cols if f"SIG{c}" in ds.columns]
    drop = intensity_cols + sigi_cols
    if drop:
        ds = ds.drop(columns=drop)

    ds.cell = gemmi.read_structure(str(pdb_path)).cell

    # Named to end in 'data.mtz' so the GUI result-viewer's exclusion glob skips
    # it and only picks up the refined output mtz.
    input_mtz = Path(out_dir) / "input_data.mtz"
    ds.write_mtz(str(input_mtz))
    return str(input_mtz), anomalous


def run(mtz_path, pdb_path, out_dir, device="cpu", macro_cycles=MACRO_CYCLES):
    import torch
    from torchref import LBFGSRefinement
    from torchref.model.parameter_wrappers import PositiveMixedTensor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_mtz, anomalous = prepare_data(mtz_path, pdb_path, out_dir)
    print(f"anomalous={anomalous}  device={device}", flush=True)

    dev = torch.device(device)
    ref = LBFGSRefinement(
        data_file=input_mtz,
        pdb=str(pdb_path),
        device=dev,
        target_mode="ml",  # maximum-likelihood x-ray target
    )

    # Reset every atomic B-factor to a flat starting value and invalidate the
    # structure-factor cache so the new values take effect.
    adp = ref.model.adp
    ref.model.adp = PositiveMixedTensor(
        torch.full_like(adp().detach(), RESET_B),
        refinable_mask=adp.refinable_mask,
        name="adp",
    )
    ref.model.reset_cache()

    rw0, rf0 = ref.get_rfactor()
    print(
        f"Initial (B reset to {RESET_B}): Rwork={rw0:.4f}  Rfree={rf0:.4f}",
        flush=True,
    )

    # Macrocycles of bulk-solvent + scaling followed by atomic ADP refinement.
    for cycle in range(macro_cycles):
        ref.get_scales()
        ref.refine_adp()
        rw, rf = ref.get_rfactor()
        print(
            f"Cycle {cycle + 1:2d}/{macro_cycles}: Rwork={rw:.4f}  Rfree={rf:.4f}",
            flush=True,
        )

    ref.write_out_pdb(str(out_dir / "refined.pdb"))
    # torchref writes refined amplitudes + 2mFo-DFc/mFo-DFc map coefficients;
    # anomalous content is carried by the signed HKLs already in the data.
    ref.write_out_mtz(str(out_dir / "refined.mtz"))
    print(f"wrote refined.pdb and refined.mtz to {out_dir}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="torchref refinement worker")
    p.add_argument("--mtz", required=True, help="abismal per-epoch MTZ to refine against")
    p.add_argument("--pdb", required=True, help="starting model")
    p.add_argument("--out-dir", required=True, help="directory for refined outputs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--macro-cycles", type=int, default=MACRO_CYCLES)
    args = p.parse_args(argv)
    run(
        args.mtz,
        args.pdb,
        args.out_dir,
        device=args.device,
        macro_cycles=args.macro_cycles,
    )


if __name__ == "__main__":
    main()
