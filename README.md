# abismal
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