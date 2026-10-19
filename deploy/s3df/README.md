# abismal on S3DF Jupyter

A demo of the abismal notebook GUI in S3DF's Jupyter (Open OnDemand) app, for users who cannot install
software in the shared, fast storage where applications belong. A maintainer builds one shared, read-only
environment; each user's Jupyter session sources a short pre-launch script that points at it.

Longer term, the same environment could be built into an image hosted by the S3DF admins, and the
pre-launch step would go away.

## For the maintainer: build the shared environment

On a node with internet access, as someone who can write to the install location:

```bash
git clone -b s3df-demo https://github.com/rs-station/abismal
abismal/deploy/s3df/install_env.sh /sdf/group/<facility>/sw/abismal
```

This builds `PREFIX/env` with Python 3.12, PyTorch with CUDA, `abismal[gui]`, torchref and
JupyterLab. It also stages the lysozyme example data and its reference files in
`PREFIX/share/demo/hewl`, makes everything world-readable, and ends with a smoke test.

Options:
- `--ref <git ref>`: the abismal version to install (default `torch`).
- `--torch-index <url>`: the PyTorch wheel index. The default is CUDA 12.8; pick the newest CUDA the S3DF
  GPU driver supports (`nvidia-smi` on a GPU node shows the driver's CUDA version).
- `--with-dials`: add DIALS for `.expt`/`.refl` inputs (several GB).

Rerunning updates the install in place. Then set the default `ABISMAL_S3DF_PREFIX` in
`demo_setup.sh` to the install location, so users need no settings.

## For users: start the demo

1. Open the S3DF OnDemand portal and launch the **Jupyter** app.
2. Request a GPU partition, one GPU and a few CPU cores (refinement runs on the CPUs), and your
   account.
3. In the app's custom-script field, paste one line:

   ```bash
   source <(curl -fsSL https://raw.githubusercontent.com/rs-station/abismal/s3df-demo/deploy/s3df/demo_setup.sh)
   ```

   or, without needing internet access from the compute node:

   ```bash
   source /sdf/group/<facility>/sw/abismal/demo_setup.sh
   ```

   The script must be **sourced**. `curl ... | bash` runs it in a separate shell, and Jupyter would
   start without the environment.
4. When JupyterLab opens, go to `~/abismal-demo`, open `abismal_s3df_demo.ipynb` and run the cells in order.

The script writes only to your own space:
- your copy of the notebook in `~/abismal-demo`;
- a link to the shared example data;
- compile caches in your scratch directory (or `~/.cache` without one);
- a Jupyter kernel named *abismal (S3DF)*.

It never overwrites an edited notebook. To put things elsewhere, set `ABISMAL_DEMO_DIR`,
`ABISMAL_CACHE_DIR` or `ABISMAL_S3DF_PREFIX` before the source line.

## Troubleshooting

- **"no shared install at …" in the session log**: `ABISMAL_S3DF_PREFIX` is wrong, or the install is missing.
- **The GUI appears as text**: the session's Jupyter has no widget support, so the shared environment
  did not start it. Check that the custom script ran (see the session's output log), and choose the
  *abismal (S3DF)* kernel.
- **"No GPU in this session"**: the app was launched without a GPU; relaunch on a GPU partition.
- **The first run pauses for a minute or two before training**: that is `torch.compile` building
  kernels. They are cached in your scratch directory, so later runs start faster.
