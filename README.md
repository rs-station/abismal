# abismal
**A**pproximate **B**ayesian **I**nference for **S**caling and **M**erging at **A**dvanced **L**ightsources

Scaling and merging for large diffraction datasets using stochastic variational inference and deep learning.

This project is under development. 


# Installation with NVIDIA CUDA Support
```bash
conda create -yn abismal -c conda-forge dials
conda activate abismal
pip install --upgrade pip
pip install abismal[cuda]@git+https://github.com/rs-station/abismal.git
```
Test the installation with
```bash
abismal --list-devices
```

