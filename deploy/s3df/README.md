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
To update, `git pull` in the clone and run the script again.

Options:
- `--torch-index <url>`: take torch from this PyTorch wheel index. Not normally needed: PyPI's torch
  brings its own CUDA (13) and needs only an NVIDIA driver of version 580 or newer. If the node's
  driver is older, the check says so; rerun with, for example,
  `--torch-index https://download.pytorch.org/whl/cu128`.
- `--with-dials`: add DIALS for `.expt`/`.refl` inputs (several GB).

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

It never overwrites an edited notebook. To put things elsewhere, set `ABISMAL_DEMO_DIR` or
`ABISMAL_CACHE_DIR` before the source line.

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
