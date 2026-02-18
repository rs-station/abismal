# abismal
![Build](https://github.com/rs-station/abismal/workflows/Build/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/abismal?color=blue)](https://pypi.org/project/abismal/)
[![codecov](https://codecov.io/gh/rs-station/abismal/branch/main/graph/badge.svg)](https://codecov.io/gh/rs-station/abismal)

**A**pproximate **B**ayesian **I**nference for **S**caling and **M**erging at **A**dvanced **L**ightsources

Scaling and merging for large diffraction datasets using stochastic variational inference and deep learning.

This project is under development. 


# Installation
First create a conda env with dials,
```bash
conda create -yn abismal -c conda-forge dials
conda activate abismal
```
Next install abismal. 
For the CPU version, run 

```bash
pip install --upgrade pip
pip install abismal
```

For NVIDIA CUDA support, we recommend you use the anaconda python distribution. The following will create a new conda environment and install abismal:

```bash
pip install --upgrade pip
pip install abismal[cuda]
```

You can now use abismal with GPU acceleration by running `conda activate abismal`. 
You can test GPU support by typing `abismal --list-devices`. 

# Running tests
Abismal CI runs tests on each pull request. Development installs are similar to a normal install, but it is important
to make sure that you install `abismal[dev]` in a fresh environment. 
Running the following commands will set up an environment. 
```
git clone https://github.com/rs-station/abismal.git
cd abismal
conda create -yn abismal -c conda-forge dials python=3.12
conda activate abismal
pip install -e .[dev]
```

Tests are run by calling `pytest` in the root of the abismal source code directory. 



