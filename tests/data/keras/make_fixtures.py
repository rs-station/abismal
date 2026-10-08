"""
Regenerate the abismal 1.x fixtures that tests/test_convert.py converts.

This needs abismal 1.x (TensorFlow) importable, NOT the abismal in this tree:

    PYTHONPATH=/path/to/abismal-1.x python tests/data/keras/make_fixtures.py

For each fixture it runs a few training steps of the 1.x CLI on
tests/data/conventional.mtz and keeps the last epoch's .keras file. tf_keras
writes the zip uncompressed; the members are copied byte for byte into a
deflated zip, which is ten times smaller. Beside each it writes an .npz of
what 1.x computes from that file -- a fixed input batch, the standardized
inputs, the scale network's output parameters, the residual branch of each
feed forward block, the surrogate posterior's moments and the prior's Σ.

`unshared` and `gated_unshared` (share_weights=False, which the CLI never
builds) are made by editing the default fixture's config instead, with random
weights, and their references are computed from the live model, because
tf_keras cannot load those files back. `unbuilt` is the CLI's epoch_0.keras,
saved before the scale model was built.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MTZ = os.path.join(HERE, os.pardir, "conventional.mtz")
COMMON = [
    "-d", "8", "--epochs", "1", "--steps-per-epoch", "3", "--batch-size", "3",
    "--d-model", "4", "--layers", "2", "--num-cpus", "1",
]
# name: (number of copies of the mtz, extra CLI arguments)
FIXTURES = {
    "default": (1, []),
    "intensity_gamma": (1, ["--posterior-type", "intensity", "--posterior-distribution", "gamma"]),
    "intensity_lognormal": (1, ["--posterior-type", "intensity", "--posterior-distribution", "lognormal"]),
    "lognormal": (1, ["--posterior-distribution", "lognormal"]),
    "nakagami": (1, ["--posterior-distribution", "nakagami"]),
    "gated": (1, ["--gated"]),
    "auto_wilson": (1, ["--prior-distribution", "auto_wilson"]),
    "empirical_wilson": (1, ["--prior-distribution", "empirical_wilson"]),
    "mvn": (1, ["--posterior-distribution", "normal", "--posterior-rank", "3", "--prior-distribution", "normal"]),
    "scale_prior": (1, ["--optimize-scale-prior"]),
    "separate": (2, ["--separate"]),
    "double_wilson": (2, ["--separate", "--parents", "0,0", "-r", "0.0,0.99"]),
}


def deflate(src, dst):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        for info in zin.infolist():
            zout.writestr(info.filename, zin.read(info.filename))


def run_cli(workdir):
    inputs = []
    for i in range(2):
        path = os.path.join(workdir, f"input_{i}.mtz")
        shutil.copy(MTZ, path)
        inputs.append(path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for name, (copies, extra) in FIXTURES.items():
        out = os.path.join(workdir, name)
        cmd = [sys.executable, "-m", "abismal.command_line.abismal", *inputs[:copies], *COMMON, "-o", out, *extra]
        print(" ".join(cmd[1:]), flush=True)
        log = os.path.join(workdir, name + ".log")
        with open(log, "w") as f:
            # cwd: `python -m` puts it first on sys.path, ahead of PYTHONPATH,
            # and from a checkout of abismal 2 that would run the wrong abismal.
            subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=workdir)
        if not os.path.exists(os.path.join(out, "epoch_1.keras")):
            raise SystemExit(f"{name}: the 1.x CLI saved no epoch_1.keras:\n" + open(log).read()[-3000:])
        deflate(os.path.join(out, "epoch_1.keras"), os.path.join(HERE, name + ".keras"))
        if name == "default":
            deflate(os.path.join(out, "epoch_0.keras"), os.path.join(HERE, "unbuilt.keras"))


# ---------------------------------------------------------------------------
# Everything below runs under TensorFlow, in a child process.
# ---------------------------------------------------------------------------

def _tf():
    import numpy as np
    import tensorflow as tf
    import tf_keras as tfk
    import abismal
    if not os.path.exists(os.path.join(os.path.dirname(abismal.__file__), "merging", "merging.py")) \
            or "tensorflow" not in open(os.path.join(os.path.dirname(abismal.__file__), "merging", "merging.py")).read():
        raise SystemExit(f"{abismal.__file__} is not abismal 1.x; put 1.x on PYTHONPATH.")
    from abismal.merging import VariationalMergingModel  # noqa: registers the classes
    return np, tf, tfk


def make_batch(model, seed=0):
    import numpy as np
    rac = model.surrogate_posterior.rac
    rng = np.random.default_rng(seed)
    lengths = [5, 7, 4]
    n = sum(lengths)
    hunique = np.asarray(rac.Hunique)
    idx = rng.choice(len(hunique), size=n, replace=False)
    # Drawn around the moments the model standardizes with, so the network
    # sees inputs on the scale it was trained on.
    m_mean = np.reshape(model.standardize_metadata.mean.numpy(), -1)
    m_std = np.reshape(model.standardize_metadata.std.numpy(), -1)
    i_mean = float(np.reshape(model.standardize_intensity.mean.numpy(), -1)[0])
    i_std = float(np.reshape(model.standardize_intensity.std.numpy(), -1)[0])
    width = len(m_mean)
    return dict(
        asu_id=np.asarray(rac.asu_id)[idx][:, None].astype("int32"),
        hkl=hunique[idx].astype("int32"),
        resolution=np.asarray(rac.dHKL)[idx][:, None].astype("float32"),
        wavelength=np.ones((n, 1), dtype="float32"),
        metadata=(m_mean + m_std * rng.normal(size=(n, width))).astype("float32"),
        iobs=(np.abs(rng.normal(size=(n, 1))) * i_std + 0.1 * i_mean).astype("float32"),
        sigiobs=(np.abs(rng.normal(size=(n, 1))) * 0.05 * i_std + 0.01 * i_mean).astype("float32"),
        lengths=np.array(lengths, dtype="int64"),
    )


def ragged(batch):
    import tensorflow as tf
    keys = ["asu_id", "hkl", "resolution", "wavelength", "metadata", "iobs", "sigiobs"]
    return tuple(tf.RaggedTensor.from_row_lengths(batch[k], batch["lengths"]) for k in keys)


def scale_params(model, inputs):
    """ImageScaler.call up to q_params (and p_params), with no metadata noise."""
    import tensorflow as tf
    from abismal.scaling.scaling import batch_normalize
    self = model.scale_model
    asu_id, hkl, resolution, wavelength, metadata, iobs, sigiobs = inputs
    image = [metadata, iobs, sigiobs]
    if self.hkl_to_imodel:
        image.append(0.02 * tf.cast(hkl, metadata.dtype))
    image = tf.concat(image, axis=-1)
    if self.batch_normalize:
        image = tf.ragged.map_flat_values(batch_normalize, image)
    scale = image[..., :-2]
    image = tf.ragged.map_flat_values(self.image_network, self.input_image(image))
    image = self.pool(image)
    scale_in = tf.ragged.map_flat_values(self.input_scale, scale) + image
    q_latent = tf.ragged.map_flat_values(self.scale_network, scale_in)
    out = {"q_params": tf.ragged.map_flat_values(self.output_dense, q_latent).flat_values.numpy()}
    if self.optimize_prior_scale:
        out["p_params"] = tf.ragged.map_flat_values(self.output_prior, image).flat_values.numpy()
    return out


def reference(path, model=None):
    np, tf, tfk = _tf()
    if model is None:
        model = tfk.saving.load_model(path, compile=False)
    batch = make_batch(model)
    std = model.standardize_inputs(ragged(batch), training=False)
    out = dict(batch)
    out.update(scale_params(model, std))
    out["std_metadata"] = std[4].flat_values.numpy()
    out["std_iobs"] = std[5].flat_values.numpy()
    # The residual branch of every feed forward block on its own. In q_params
    # it is swamped by the skip connection (each block's last kernel starts at
    # zero), so a kernel mapped to the wrong slot could hide there.
    x = np.random.default_rng(7).normal(size=(16, model.scale_model.mlp_width)).astype("float32")
    out["ff_input"] = x
    for name in ("scale_network", "image_network"):
        net = getattr(model.scale_model, name)
        out[f"{name}_residual"] = np.stack([np.asarray(layer(x)) - x for layer in net.layers])
    q = model.surrogate_posterior.flat_distribution()
    out["q_mean"] = np.asarray(q.mean())
    out["q_stddev"] = np.asarray(q.stddev())
    rac = model.surrogate_posterior.rac
    out["Hunique"] = np.asarray(rac.Hunique)
    out["rac_asu_id"] = np.asarray(rac.asu_id)
    if hasattr(model.prior, "sigma"):
        out["prior_sigma"] = np.asarray(model.prior.sigma, dtype="float32") * np.ones(rac.asu_size, dtype="float32")
    np.savez(os.path.splitext(path)[0] + ".npz", **out)
    print("wrote", os.path.splitext(path)[0] + ".npz", flush=True)


# Models the CLI never builds, made by editing the default fixture's config.
# Their weights are random, so that every kernel is distinct: the CLI seeds
# each layer's initializer identically, which leaves a GLU block's ff1 and ff2
# nearly equal after a few steps and a swap of the two all but invisible.
SCRIPTED = {
    "unshared": {"share_weights": False},
    "gated_unshared": {"share_weights": False, "gated": True},
}


def make_scripted(workdir, name):
    import json
    np, tf, tfk = _tf()
    template = os.path.join(HERE, "default.keras")
    with zipfile.ZipFile(template) as z:
        config = json.loads(z.read("config.json"))
    config["config"]["scale_model"]["config"].update(SCRIPTED[name])
    config.pop("compile_config", None)
    model = tfk.saving.deserialize_keras_object(config)
    trained = tfk.saving.load_model(template, compile=False)
    model(ragged(make_batch(trained, seed=1)))  # build
    for part in ("surrogate_posterior", "standardize_metadata", "standardize_intensity"):
        getattr(model, part).set_weights(getattr(trained, part).get_weights())
    rng = np.random.default_rng(3)
    for v in model.scale_model.trainable_variables:
        v.assign(rng.normal(size=v.shape).astype("float32") * 0.3)
    raw = os.path.join(workdir, name + ".keras")
    model.save(raw)
    path = os.path.join(HERE, name + ".keras")
    deflate(raw, path)
    # tf_keras cannot load an unshared model back (it never builds the second
    # network), so the reference comes from the live model.
    reference(path, model=model)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--reference":
        for path in sys.argv[2:]:
            reference(path)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--scripted":
        make_scripted(sys.argv[2], sys.argv[3])
        return
    with tempfile.TemporaryDirectory() as workdir:
        run_cli(workdir)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        paths = [os.path.join(HERE, name + ".keras") for name in FIXTURES]
        subprocess.run([sys.executable, __file__, "--reference", *paths], check=True, env=env)
        for name in SCRIPTED:
            subprocess.run([sys.executable, __file__, "--scripted", workdir, name], check=True, env=env)


if __name__ == "__main__":
    main()
