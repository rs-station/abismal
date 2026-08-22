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

For NVIDIA CUDA support, add the `cuda` extra:

```bash
pip install "abismal[cuda]"
```

Test GPU support with `abismal --list-devices`.

## Per-epoch refinement (`--torchref-pdb`)

Add the `torchref` extra. It pulls PyTorch, so it is optional -- a plain merging
run does not need it. Extras combine, so ask for both in one command rather than
installing them one after another:

```bash
pip install "abismal[cuda,torchref]"
```

torchref then lives in the same interpreter as abismal, and `--torchref-pdb`
works with no further configuration. Refinement still runs in a subprocess, so a
failure there cannot take training down, but that subprocess is the environment
you are already in -- there is no second environment to build or point at.

Both extras bring their own GPU stack and they do not collide: TensorFlow pins
the CUDA 12 wheels (`nvidia-cudnn-cu12==9.3.0.75`) while current PyTorch pins the
CUDA 13 ones (`nvidia-cudnn-cu13`). Different package names, so each framework
gets exactly the version it asked for. Older PyTorch (2.9 and earlier) was also
built on CUDA 12 and did contend with TensorFlow for those pins -- it resolved to
the newer versions and both still worked, but the combination above is cleaner.

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
pip install -e ".[dev,torchref]"
```

Tests are run by calling `pytest` in the root of the abismal source code directory. 



