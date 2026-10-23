# abismal on S3DF Jupyter

A demo of the abismal notebook GUI in S3DF's Jupyter (Open OnDemand) app, for users who cannot install
software in the shared, fast storage where applications belong. A maintainer clones this repository
there and builds one environment inside the clone; each user's Jupyter session sources a short
pre-launch script from the same clone.

Longer term, the same environment could be built into an image hosted by the S3DF admins, and the
pre-launch step would go away.

## For the maintainer: build the environment

On a node with internet access (a GPU node, if possible, so the closing check can see the GPU), as
someone who can write to the shared location:

```bash
git clone -b s3df-demo https://github.com/rs-station/abismal /sdf/group/<facility>/abismal
/sdf/group/<facility>/abismal/deploy/s3df/install_env.sh
```

This builds `deploy/s3df/install/` inside the clone (git-ignored): Python 3.12, PyTorch with CUDA,
abismal with its GUI and torchref installed from the clone itself, and JupyterLab. It also stages the
lysozyme example data and its reference files, makes the clone world-readable, and ends with a check.
abismal is installed in editable mode, so `git pull` in the clone updates it; rerun the script only
when dependencies change.

Options:
- `--torch-index <url>`: the PyTorch wheel index to take torch from. The default is the CUDA 12.9
  builds (`https://download.pytorch.org/whl/cu129`), which S3DF's GPU driver supports; they bring
  their own CUDA libraries, so no CUDA toolkit is needed, and they also run on newer drivers. Pass
  `pypi` for PyPI's default build (CUDA 13, driver 580 or newer). If torch cannot use the node's GPU,
  the closing check says so.
- `--without-dials`: leave out DIALS, which reads `.expt`/`.refl` inputs and is installed by default
  (several GB). `.mtz` and CrystFEL `.stream` inputs never need it. Rerunning the script without
  this flag adds DIALS to an environment built without it.

## For users: start the demo

1. Open the S3DF OnDemand portal and launch the **Jupyter** app.
2. Request a GPU partition, one GPU and a few CPU cores (refinement runs on the CPUs), and your
   account.
3. In the app's custom-script field, paste one line:

   ```bash
   source /sdf/group/<facility>/abismal/deploy/s3df/demo_setup.sh
   ```

   It must be **sourced** from the shared clone: it finds the environment next to itself, and
   `bash demo_setup.sh` would run it in a separate shell, so Jupyter would start without the
   environment. It needs no internet access.
4. When JupyterLab opens, go to `~/abismal-demo`, open `abismal_s3df_demo.ipynb` and run the cells in order.

The script writes only to your own space:
- your copy of the notebook in `~/abismal-demo`;
- a link to the shared example data;
- compile caches in your scratch directory (or `~/.cache` without one);
- a Jupyter kernel named *abismal (S3DF)*.

JupyterLab opens on the demo notebook, with the file browser in `~/abismal-demo`, whatever was open
in earlier sessions; demo sessions keep their layout in `~/abismal-demo/.jupyter-workspaces`, so the
user's own JupyterLab layout is neither restored nor changed (`jupyter/abismal_demo_layout.py`).

It never overwrites an edited notebook. To put things elsewhere, set `ABISMAL_DEMO_DIR` or
`ABISMAL_CACHE_DIR` before the source line.

## For users: run abismal your own way

To use the shared environment without the demo, for the command line, batch jobs, or the GUI in a
Jupyter session of your own, source `setup.sh` instead:

```bash
source /sdf/group/<facility>/abismal/deploy/s3df/setup.sh
```

Put it in a terminal, at the top of a batch script, in `~/.bashrc`, or in the Jupyter app's
custom-script field. Like `demo_setup.sh`, it must be sourced, not run. It:
- puts the environment first on `PATH`, so `abismal`, `abismal.gui`, `abismal.cchalf`, `python` and
  `jupyter` come from it;
- preloads the environment's own C++ runtime (see Troubleshooting);
- sends compile caches to your scratch directory (or `~/.cache` without one);
- registers the *abismal (S3DF)* Jupyter kernel, so any Jupyter can run the GUI notebook. Set
  `ABISMAL_REGISTER_KERNEL=0` before sourcing to skip this.

It creates no demo directory and leaves JupyterLab's layout alone. Then, for example:

```bash
abismal --help
abismal -o run1 --anomalous -d 1.8 data.mtz     # a merge on the command line
abismal.gui                                      # the GUI notebook in a new JupyterLab
```

In a Slurm batch job, request a GPU and source the script before calling `abismal`:

```bash
#!/bin/bash
#SBATCH --partition=<gpu partition> --gpus=1 --cpus-per-task=4 --account=<account>
source /sdf/group/<facility>/abismal/deploy/s3df/setup.sh
abismal -o run1 --anomalous -d 1.8 data.mtz
```

## Troubleshooting

- **"no environment at …" in the session log**: the path in the source line is not the clone the
  installer was run in, or the installer has not been run.
- **The GUI appears as text**: the session's Jupyter has no widget support, so the shared environment
  did not start it. Check that the custom script ran (see the session's output log), and choose the
  *abismal (S3DF)* kernel.
- **"No GPU in this session"**: the app was launched without a GPU; relaunch on a GPU partition. If
  the session has a GPU and torch still cannot use it, the driver is older than torch's CUDA build;
  see `--torch-index` above.
- **The first run pauses for a minute or two before training**: that is `torch.compile` building
  kernels. They are cached in your scratch directory, so later runs start faster.
