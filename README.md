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
micromamba create -yn abismal -c conda-forge python=3.12 dials "numpy<2.4" "pandas<2.4" "scipy<1.18"
micromamba activate abismal
pip install --upgrade pip
pip install abismal
```

`conda` works in place of `micromamba` throughout.

abismal trains with [PyTorch](https://pytorch.org), which pip installs with it.
If torch is already installed -- on Colab, or in an environment you built for a
particular CUDA version -- abismal uses that one and leaves it alone.

## GPUs

There is nothing extra to install for a GPU. On Linux the default torch wheel
bundles CUDA, and on Apple silicon torch includes the Metal (MPS) backend.
abismal picks a device for itself: the CUDA GPU given by `--gpu-id` (0 by
default) if there is one, otherwise MPS, otherwise the CPU. `--device` names one
explicitly (`cpu`, `cuda:1`, `mps`). See what torch can see with

```bash
abismal --list-devices
```

On a machine with no GPU, the CUDA build is several gigabytes that never get
used. Take the CPU build instead by adding PyTorch's CPU index to the install:

```bash
pip install abismal --extra-index-url https://download.pytorch.org/whl/cpu
```

## Extras

abismal ships these optional extras. They **compose** -- name them in one set of
brackets rather than running separate installs, so pip resolves everything in a
single pass:

| extra | what it adds |
| --- | --- |
| `torchref` | torchref, for per-epoch refinement (`--torchref-pdb`) |
| `gui` | the widget stack behind the `abismal.gui` notebook |
| `lab` | JupyterLab, to run that notebook locally |

```bash
pip install "abismal[torchref,gui,lab]"
```

abismal 1.x had a `cuda` extra for GPU TensorFlow. It is gone: torch needs no
extra for the GPU. pip warns about an extra that does not exist and installs
the rest, so old commands still work.

## Per-epoch refinement (`--torchref-pdb`)

The `torchref` extra puts torchref in the same interpreter as abismal, so
`--torchref-pdb` works with no further configuration. Refinement still runs in a
subprocess, so a failure there cannot take training down, but that subprocess is
the environment you are already in -- there is no second environment to build or
point at. It runs on the CPU, leaving the GPU to training.

## The notebook GUI

The notebook ships inside the package, and `abismal.gui` puts a copy in
whatever directory you are standing in and opens it:

```bash
pip install "abismal[gui,lab]"
cd /path/to/your/data
abismal.gui
```

`gui` is the widget stack the notebook needs on any frontend; `lab` adds
JupyterLab to run it locally. On Colab, where the frontend is already there,
`gui` alone is enough.

The notebook is one cell: it installs abismal if it is not already there, then

```python
from abismal.gui import launch

launch()
```

`launch()` builds the form and returns it for the notebook to display. It also
checks that the code running is the code you think it is -- a module loaded at a
different version than is installed cannot be swapped without a restart, and it
says so rather than rendering nothing.

On Colab, `launch(example_data=True)` additionally fetches a lysozyme SSAD
dataset and its reference model, and adds torchref, so there is something to
merge and refine straight away. It is off by default: bring your own data and
nothing is downloaded.

`ArgparseGUI().to_widget()` still builds the form alone, for anyone who wants it
without the setup around it.

That directory is where the file pickers open and what a relative `out_dir`
resolves against, so **run it from your data**. The copy is yours: a second
`abismal.gui` in the same directory opens what is already there rather than
overwriting your edits, and `--fresh` replaces it with the packaged one.
Anything the launcher does not recognise is handed to `jupyter lab`, so
`abismal.gui --no-browser --port 9999` works as you would expect.

Nothing needs to be copied or symlinked by hand, and the notebook does not have
to sit with your data: the form asks the running server which directory it was
launched from rather than looking at the kernel's own, which jupyter sets to
wherever the .ipynb happens to be.

`out_dir` may be anywhere you can write, including another filesystem. The 3D
viewer embeds the model and maps in its own page rather than fetching them from
the server, so the server's root directory does not constrain it either.

On Colab, upload your files to the session first (the Files pane, or
`google.colab.files.upload()`) and point the file browser at `/content`. Colab
already has torch, so installing abismal there does not download it again.

## About the version pins

abismal itself pins nothing tightly. The pins in the recipe above come from
torchref, which declares

| package | torchref allows |
| --- | --- |
| numpy | `>=2.0, <2.4` |
| pandas | `>=2.0, <2.4` |
| scipy | `>=1.10, <1.18` |
| torch | `>=2.4, <2.14` |

Left alone, conda solves DIALS onto numpy 2.5, and adding torchref afterwards
makes pip downgrade conda-built packages underneath DIALS.
Constraining the conda solve to torchref's windows avoids that, at the cost of
a slightly older DIALS (3.26 rather than 3.30 at the time of writing). Without
torchref the pins are harmless, and they let you add it later. Re-check them
when torchref moves its ceilings.

# Running tests
Abismal CI runs tests on each pull request. Development installs are similar to a normal install, but it is important
to make sure that you install `abismal[dev]` in a fresh environment. 
Running the following commands will set up an environment. 
```
git clone https://github.com/rs-station/abismal.git
cd abismal
micromamba create -yn abismal -c conda-forge python=3.12 dials "numpy<2.4" "pandas<2.4" "scipy<1.18"
micromamba activate abismal
pip install -e ".[dev,torchref,gui]" --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index gives the CPU build of torch, which is all the tests need; drop
it to test on a GPU. TensorFlow is not a dependency, and CI checks it stays
that way.

Tests are run by calling `pytest` in the root of the abismal source code directory. 



