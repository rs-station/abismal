# abismal
![Build](https://github.com/rs-station/abismal/workflows/Build/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/abismal?color=blue)](https://pypi.org/project/abismal/)
[![codecov](https://codecov.io/gh/rs-station/abismal/branch/main/graph/badge.svg)](https://codecov.io/gh/rs-station/abismal)

**A**pproximate **B**ayesian **I**nference for **S**caling and **M**erging at **A**dvanced **L**ightsources

Scaling and merging for large diffraction datasets using stochastic variational inference and deep learning.

This project is under development. 


# Installation

abismal, DIALS and torchref all install into a single environment. DIALS comes
from conda-forge; everything else comes from pip on top of it.

```bash
micromamba create -yn abismal -c conda-forge python=3.12 dials "pandas<2.4" "scipy<1.18"
micromamba activate abismal
pip install --upgrade pip
pip install abismal
```

`conda` works in place of `micromamba` throughout.

## Extras

abismal ships two optional extras. They **compose** -- name both in one set of
brackets rather than running two installs, so pip resolves everything in a single
pass:

| extra | what it adds |
| --- | --- |
| `cuda` | GPU-accelerated TensorFlow for merging (`tensorflow[and-cuda]`) |
| `torchref` | PyTorch + torchref, for per-epoch refinement (`--torchref-pdb`) |

```bash
pip install "abismal[cuda,torchref]"
```

Test GPU support with `abismal --list-devices`.

## Per-epoch refinement (`--torchref-pdb`)

The `torchref` extra puts torchref in the same interpreter as abismal, so
`--torchref-pdb` works with no further configuration. Refinement still runs in a
subprocess, so a failure there cannot take training down, but that subprocess is
the environment you are already in -- there is no second environment to build or
point at.

**Install the CPU build of PyTorch.** Refinement runs on CPU -- that is the
default and nothing exposes a GPU option -- so the CUDA build of torch is several
gigabytes that never get used, and it puts a second CUDA stack next to
TensorFlow's for no benefit. Install torch from PyTorch's CPU index *first*, then
the extras:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "abismal[cuda,torchref]"
```

Order matters. The second command finds `torch>=2.4.0` already satisfied and
leaves it alone. Doing it the other way around installs the CUDA build and then
you are stuck with it: **do not try to uninstall it afterwards.** PyTorch's CUDA
wheels share the `nvidia/` namespace with TensorFlow's, and removing them deletes
shared libraries that TensorFlow needs -- it will report no GPU and silently fall
back to CPU. Rebuild the environment instead.

If you do want the CUDA build of torch, it coexists with `abismal[cuda]` as long
as the two sit on different CUDA majors: TensorFlow 2.18 pins the CUDA 12 wheels
(`nvidia-cudnn-cu12==9.3.0.75`) and current PyTorch pins the CUDA 13 ones, so
each gets exactly what it asked for. Older PyTorch (2.9 and earlier) was also
built on CUDA 12 and contended with TensorFlow for those pins; it resolved to the
newer versions and both still worked, but nothing guarantees that in general.
Re-check `pip list | grep nvidia` after any TensorFlow or PyTorch bump.

## About the version pins

The three tool-chains agree on exactly one numpy window, and the recipe above is
built to land inside it:

| component | numpy |
| --- | --- |
| tensorflow 2.18 | `>=1.26, <2.1` |
| torchref | `>=2.0, <2.4` |
| conda-forge dxtbx | `>=2.3.4` (declared) |

`pip install abismal` pulls numpy down to 2.0.2 to satisfy TensorFlow, below the
floor conda-forge declares for dxtbx. That combination is tested -- the full
suite, DIALS tests included, passes on it -- but it is why the pins exist:

- `pandas<2.4` / `scipy<1.18` keep the conda solve inside torchref's ceilings, so
  pip does not have to overwrite conda-built packages later.
- `scikit-image<0.25` (a dependency of abismal, pinned in `pyproject.toml`) keeps
  `tifffile` off releases that require numpy>=2.1.

TensorFlow 2.20 dropped its numpy ceiling, which would let all three agree on
numpy 2.3.x with no override at all. abismal is pinned to 2.18 for now.

# Running tests
Abismal CI runs tests on each pull request. Development installs are similar to a normal install, but it is important
to make sure that you install `abismal[dev]` in a fresh environment. 
Running the following commands will set up an environment. 
```
git clone https://github.com/rs-station/abismal.git
cd abismal
micromamba create -yn abismal -c conda-forge python=3.12 dials "pandas<2.4" "scipy<1.18"
micromamba activate abismal
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,cuda,torchref]"
```

Tests are run by calling `pytest` in the root of the abismal source code directory. 



