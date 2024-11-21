# abismal
**A**pproximate **B**ayesian **I**nference for **S**caling and **M**erging at **A**dvanced **L**ightsources

Scaling and merging for large diffraction datasets using stochastic variational inference and deep learning.

This project is under development. 


# Installation with NVIDIA CUDA Support
```bash
conda create -yn abismal -c conda-forge dials
conda activate dials
pip install --upgrade pip
pip install abismal[cuda]@git+https://github.com/rs-station/abismal.git
```
Test the installation with
```bash
abismal --list-devices
```

For NVIDIA CUDA support, we recommend you use the anaconda python distribution. The following will create a new conda environment and install abismal:

```bash
source <(curl -s https://raw.githubusercontent.com/rs-station/abismal/main/install.sh)
```

You can now use abismal with GPU acceleration by running `conda activate abismal`

