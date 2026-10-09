"""
Time abismal's training step, separately from data loading.

    python devtools/bench/bench_step.py                     # synthetic batches
    python devtools/bench/bench_step.py --input data.mtz --dmin 1.7 --anomalous
    python devtools/bench/bench_step.py --device cpu --d-model 32 --layers 20

Reports milliseconds per step for the eager model and for torch.compile, the
compile time, and how long the data pipeline takes to produce a batch on its
own, so a slow run can be pinned on the model or on the input. Batches are
moved to the device once up front; the steps reuse them.
"""
import argparse
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="*", default=None, help="Reflection files. Synthetic data if omitted.")
    p.add_argument("--dmin", type=float, default=2.0)
    p.add_argument("--anomalous", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=100, help="Images per batch")
    p.add_argument("--reflections-per-image", type=int, default=200, help="Synthetic data only")
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def synthetic_images(rac, n_images, n_refls, metadata_length=5, seed=0):
    rng = np.random.default_rng(seed)
    rasu = rac.reciprocal_asus[0]
    images = []
    for _ in range(n_images):
        n = rng.poisson(n_refls) + 1
        H = rasu.Hunique[rng.choice(rasu.asu_size, n)]
        I = rng.gamma(1., 100., size=(n, 1)).astype(np.float32)
        images.append((
            np.zeros((n, 1), np.int32), H.astype(np.int32),
            np.ones((n, 1), np.float32), np.ones((n, 1), np.float32),
            rng.normal(size=(n, metadata_length)).astype(np.float32),
            I, np.sqrt(I + 10.).astype(np.float32),
        ))
    return images


def build(args, metadata_length, cell, spacegroup):
    from abismal.symmetry import ReciprocalASU, ReciprocalASUGraph
    from abismal.prior.structure_factor.wilson import WilsonPrior
    from abismal.surrogate_posterior.structure_factor import FoldedNormalPosterior
    from abismal.scaling import ImageScaler
    from abismal.likelihood import StudentTLikelihood
    from abismal.merging import VariationalMergingModel

    rac = ReciprocalASUGraph(ReciprocalASU(cell, spacegroup, args.dmin, anomalous=args.anomalous))
    prior = WilsonPrior(rac)
    loc = prior.flat_distribution().mean
    q = FoldedNormalPosterior(rac, loc_init=loc, scale_init=prior.flat_distribution().stddev)
    scale = ImageScaler(
        metadata_length, mlp_width=args.d_model, mlp_depth=args.layers,
        hidden_units=2 * args.d_model, prior_name="lognormal", pre_activation="relu",
        batch_normalize=False, precision=getattr(args, "precision", "float32"),
    )
    return VariationalMergingModel(
        scale, q, prior, StudentTLikelihood(32.), mc_samples=args.mc_samples, standardize=True,
    )


def time_steps(trainer, batches, steps, device):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    sync()
    start = time.perf_counter()
    for i in range(steps):
        trainer.train_step(batches[i % len(batches)])
    sync()
    return (time.perf_counter() - start) / steps * 1e3


def main():
    args = parse_args()
    import gemmi
    from abismal.device import select_device, configure
    from abismal.io import Batch
    from abismal.optimizers import Adam
    from abismal.training import Trainer, parameter_groups

    device = select_device(0, args.device)
    configure(device)

    if args.input:
        from abismal.io.manager import DataManager
        dm = DataManager(args.input, dmin=args.dmin, batch_size=args.batch_size, num_cpus=4)
        train, _ = dm.get_train_test_splits()
        start = time.perf_counter()
        it = iter(train)
        cpu_batches = [next(it) for _ in range(20)]
        pipeline_ms = (time.perf_counter() - start) / 20 * 1e3
        cell, spacegroup = dm.cell, dm.spacegroup
        metadata_length = cpu_batches[0].metadata.shape[-1]
    else:
        cell = gemmi.UnitCell(79.3, 79.3, 37.8, 90., 90., 90.)
        spacegroup = gemmi.SpaceGroup("P 43 21 2")
        pipeline_ms = None
        metadata_length = 5

    model = build(args, metadata_length, cell, spacegroup).to(device)
    if not args.input:
        images = synthetic_images(model.surrogate_posterior.rac, 20 * args.batch_size, args.reflections_per_image)
        cpu_batches = [Batch.collate(images[i:i + args.batch_size]) for i in range(0, len(images), args.batch_size)]

    batches = [b.map(lambda a: torch.as_tensor(a).to(device)) for b in cpu_batches]
    refls = np.mean([len(b.iobs) for b in cpu_batches])
    params = sum(p.numel() for p in model.parameters())
    print(f"device {device}  d_model {args.d_model}  layers {args.layers}  mc {args.mc_samples}")
    print(f"{refls:.0f} reflections per batch, {model.surrogate_posterior.rac.asu_size} unique reflections, {params} parameters")
    if pipeline_ms is not None:
        print(f"data pipeline (first batches, includes parsing): {pipeline_ms:.1f} ms/batch")

    trainer = Trainer(model, Adam(parameter_groups(model)), device=device, verbose=0)
    time_steps(trainer, batches, 5, device)
    print(f"eager:   {time_steps(trainer, batches, args.steps, device):7.2f} ms/step")

    if not args.no_compile:
        compiled = Trainer(
            model, Adam(parameter_groups(model)), device=device, verbose=0,
            compile={"dynamic": True},
        )
        start = time.perf_counter()
        time_steps(compiled, batches, 3, device)
        print(f"compile: {time.perf_counter() - start:7.1f} s for the first 3 steps")
        print(f"compiled:{time_steps(compiled, batches, args.steps, device):7.2f} ms/step")


if __name__ == "__main__":
    main()
