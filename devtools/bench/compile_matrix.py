"""
Compare torch.compile configurations for abismal's training step.

    python devtools/bench/compile_matrix.py --input data.mtz --dmin 1.7 --anomalous
    python devtools/bench/compile_matrix.py --input data.mtz --dmin 1.7 --anomalous --variants eager default

Every variant runs in its own subprocess, so compile caches and allocator state
do not carry from one to the next. Each trains a freshly built model on the same
batches, read once from the real data (their reflection counts vary, as in a
real run), and reports:

  warmup s      time for the first `--warmup` steps, which includes compiling
  ms/step       median over the following `--steps` steps
  peak alloc    torch.cuda.max_memory_allocated during the timed steps
  reserved      torch.cuda.memory_reserved at the end
  recompiles    how many times dynamo recompiled the model
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

VARIANTS = {
    "eager": {},
    "default": {"compile": {}},
    "dynamic": {"compile": {"dynamic": True}},
    "autotune": {"compile": {"mode": "max-autotune-no-cudagraphs"}},
    "autotune-dynamic": {"compile": {"mode": "max-autotune-no-cudagraphs", "dynamic": True}},
    "cudagraphs": {"compile": {"mode": "reduce-overhead"}},
    "cudagraphs-dynamic": {"compile": {"mode": "reduce-overhead", "dynamic": True}},
    "bf16-eager": {"precision": "bfloat16"},
    "bf16-default": {"precision": "bfloat16", "compile": {}},
    "bf16-autotune": {"precision": "bfloat16", "compile": {"mode": "max-autotune-no-cudagraphs"}},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--dmin", type=float, required=True)
    p.add_argument("--anomalous", action="store_true")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument("--batches", type=int, default=60, help="Distinct batches to cycle through")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    p.add_argument("--cache", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def load_batches(args, path):
    """Read the batches once and save them, so every variant sees the same ones."""
    from abismal.io.manager import DataManager
    dm = DataManager(args.input, dmin=args.dmin, batch_size=args.batch_size, num_cpus=4,
                     steps_per_epoch=args.batches, shuffle_buffer_size=10_000)
    train, _ = dm.get_train_test_splits()
    it = iter(train)
    batches = [next(it) for _ in range(args.batches)]
    np.savez(path, n=len(batches), cell=list(dm.cell.parameters), spacegroup=dm.spacegroup.xhm(),
             **{f"{i}_{k}": v for i, b in enumerate(batches) for k, v in zip(b._fields, b) if k != "num_images"},
             **{f"{i}_num_images": b.num_images for i, b in enumerate(batches)})


def run_one(args, variant):
    import gemmi
    import torch
    import torch._dynamo
    from abismal.io import Batch
    from abismal.device import configure
    from abismal.optimizers import Adam
    from abismal.training import Trainer, parameter_groups
    sys.path.insert(0, os.path.dirname(__file__))
    from bench_step import build

    device = torch.device("cuda")
    configure(device)
    data = np.load(args.cache, allow_pickle=False)
    cell = gemmi.UnitCell(*data["cell"])
    sg = gemmi.SpaceGroup(str(data["spacegroup"]))
    fields = [f for f in Batch._fields if f != "num_images"]
    batches = []
    for i in range(int(data["n"])):
        b = Batch(*[data[f"{i}_{f}"] for f in fields], int(data[f"{i}_num_images"]))
        batches.append(b.map(lambda a: torch.as_tensor(a).to(device)))

    torch.manual_seed(0)
    options = VARIANTS[variant]
    args.precision = options.get("precision", "float32")
    model = build(args, batches[0].metadata.shape[-1], cell, sg).to(device)
    trainer = Trainer(model, Adam(parameter_groups(model)), device=device, verbose=0,
                      compile=options.get("compile"))

    def sync():
        torch.cuda.synchronize()

    torch._dynamo.utils.counters.clear()
    start = time.perf_counter()
    for i in range(args.warmup):
        trainer.train_step(batches[i % len(batches)])
    sync()
    warmup = time.perf_counter() - start

    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(args.steps):
        b = batches[(args.warmup + i) % len(batches)]
        t = time.perf_counter()
        trainer.train_step(b)
        sync()
        times.append(time.perf_counter() - t)
    recompiles = sum(torch._dynamo.utils.counters["stats"].get(k, 0) for k in ("unique_graphs",))
    result = {
        "variant": variant,
        "warmup s": warmup,
        "ms/step": 1e3 * float(np.median(times)),
        "p90 ms": 1e3 * float(np.percentile(times, 90)),
        "peak alloc GB": torch.cuda.max_memory_allocated() / 1e9,
        "reserved GB": torch.cuda.memory_reserved() / 1e9,
        "graphs": recompiles,
    }
    print("RESULT " + json.dumps(result))


def main():
    args = parse_args()
    if args.run_one:
        return run_one(args, args.run_one)

    cache = os.path.abspath("compile_matrix_batches.npz")
    load_batches(args, cache)
    sizes = [len(np.load(cache)[f"{i}_iobs"]) for i in range(args.batches)]
    print(f"{args.batches} batches, {np.min(sizes)}-{np.max(sizes)} reflections")

    rows = []
    for variant in args.variants:
        cmd = [sys.executable, __file__, "--run-one", variant, "--cache", cache] + sys.argv[1:]
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not lines:
            print(f"{variant}: FAILED\n{out.stderr[-2000:]}")
            continue
        r = json.loads(lines[-1][len("RESULT "):])
        rows.append(r)
        print(f"{variant:20s} warmup {r['warmup s']:6.1f} s  {r['ms/step']:6.2f} ms/step (p90 {r['p90 ms']:6.2f})"
              f"  peak {r['peak alloc GB']:5.2f} GB  reserved {r['reserved GB']:5.2f} GB  graphs {r['graphs']}", flush=True)
    os.remove(cache)


if __name__ == "__main__":
    main()
