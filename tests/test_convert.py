"""
abismal.convert against models written by abismal 1.x.

tests/data/keras holds .keras files saved by the TensorFlow version of abismal
(short CLI runs on conventional.mtz) and, beside each, an .npz of what that
version computed from it: the surrogate posterior's moments, the prior's Σ, and
the scale network's output parameters for a fixed batch. The converted model
must reproduce them. tests/data/keras/make_fixtures.py regenerates both.
"""
import glob
import json
import os
import warnings
import zipfile

import numpy as np
import pytest
import torch

from abismal.convert import ConversionError, convert_keras, load_keras, main, read_keras
from abismal.io.batch import Batch
from abismal.saving import load_model, load_optimizer
from abismal.scaling.scaling import batch_normalize

DATA = os.path.join(os.path.dirname(__file__), "data", "keras")
FIXTURES = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(DATA, "*.npz"))
)
RTOL, ATOL = 1e-5, 1e-6


def fixture(name):
    return os.path.join(DATA, name + ".keras")


def reference(name):
    return dict(np.load(os.path.join(DATA, name + ".npz")))


def batch_from(ref):
    lengths = ref["lengths"]
    image_id = np.repeat(np.arange(len(lengths)), lengths)
    return Batch(
        torch.as_tensor(ref["asu_id"]),
        torch.as_tensor(ref["hkl"]),
        torch.as_tensor(ref["resolution"]),
        torch.as_tensor(ref["wavelength"]),
        torch.as_tensor(ref["metadata"]),
        torch.as_tensor(ref["iobs"]),
        torch.as_tensor(ref["sigiobs"]),
        torch.as_tensor(image_id),
        len(lengths),
    )


def scale_params(model, batch):
    """ImageScaler.forward up to its distribution parameters, without noise."""
    self = model.scale_model
    image = [batch.metadata, batch.iobs, batch.sigiobs]
    if self.hkl_to_imodel:
        image.append(0.02 * batch.hkl.to(batch.metadata.dtype))
    image = torch.cat(image, dim=-1)
    if self.batch_normalize:
        image = batch_normalize(image)
    scale = image[..., :self.metadata_length]
    image = self.image_network(self.input_image(image))
    image = self.pool(image, batch.image_id, batch.num_images)
    q_latent = self.scale_network(self.input_scale(scale) + image)
    out = {"q_params": self.output_dense(q_latent)}
    if self.optimize_prior_scale:
        out["p_params"] = self.output_prior(image)
    return out


def close(actual, expected):
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=RTOL, atol=ATOL,
    )


def test_fixtures_present():
    assert len(FIXTURES) >= 10


@pytest.mark.parametrize("name", FIXTURES)
def test_parity(name):
    ref = reference(name)
    model, _ = load_keras(fixture(name))
    model.eval()
    with torch.no_grad():
        rac = model.surrogate_posterior.rac
        np.testing.assert_array_equal(rac.Hunique.numpy(), ref["Hunique"])
        np.testing.assert_array_equal(rac.asu_id.numpy(), ref["rac_asu_id"])

        q = model.surrogate_posterior.flat_distribution()
        close(q.mean, ref["q_mean"])
        close(q.stddev, ref["q_stddev"])

        if "prior_sigma" in ref:
            close(model.prior.sigma * torch.ones(rac.asu_size), ref["prior_sigma"])

        batch = model.standardize_inputs(batch_from(ref), training=False)
        close(batch.metadata, ref["std_metadata"])
        close(batch.iobs, ref["std_iobs"])

        # Each block's residual branch on its own: in q_params it is swamped
        # by the skip connection, which would hide a kernel in the wrong slot.
        x = torch.as_tensor(ref["ff_input"])
        for name in ("scale_network", "image_network"):
            net = getattr(model.scale_model, name)
            residual = torch.stack([block(x) - x for block in net])
            close(residual, ref[f"{name}_residual"])

        params = scale_params(model, batch)
        close(params["q_params"], ref["q_params"])
        if "p_params" in ref:
            close(params["p_params"], ref["p_params"])

        # And the model runs end to end.
        model(batch_from(ref), mc_samples=2, training=False)


@pytest.mark.parametrize("name", ["default", "mvn", "unshared", "double_wilson"])
def test_round_trip(name, tmp_path):
    """A converted file loads with abismal.saving and matches the in-memory conversion."""
    out = convert_keras(fixture(name), str(tmp_path / f"{name}.pt"))
    model, _ = load_keras(fixture(name))
    reloaded = load_model(out)
    a, b = model.state_dict(), reloaded.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    # Fresh moments, but the optimizer the 1.x run used.
    opt = load_optimizer(out, reloaded)
    if name == "unshared":
        # Saved from a script without compiling: there is no optimizer.
        assert opt is None
        return
    assert type(opt).__name__ == "Adam"
    assert opt.state_dict()["state"] == {}


def test_shared_weights_are_tied():
    model, _ = load_keras(fixture("default"))
    assert model.scale_model.image_network is model.scale_model.scale_network


def test_unshared_networks_differ():
    model, _ = load_keras(fixture("unshared"))
    sm = model.scale_model
    assert sm.image_network is not sm.scale_network
    assert not torch.equal(sm.image_network[0].ff1.weight, sm.scale_network[0].ff1.weight)


def test_seen_and_standardization_restored():
    config, weights = read_keras(fixture("default"))
    model, _ = load_keras(fixture("default"))
    np.testing.assert_array_equal(
        model.surrogate_posterior.seen.numpy(), weights["surrogate_posterior/vars/2"],
    )
    assert model.surrogate_posterior.seen.any()
    assert bool(model.standardize_metadata.frozen) == bool(weights["standardize_metadata/vars/3"])
    assert float(model.standardize_metadata.count) == float(weights["standardize_metadata/vars/2"])


def test_default_output_path(tmp_path):
    src = tmp_path / "epoch_1.keras"
    src.write_bytes(open(fixture("default"), "rb").read())
    out = convert_keras(str(src))
    assert out == str(tmp_path / "epoch_1.pt")
    assert os.path.exists(out)


def test_cli(tmp_path, capsys):
    out = tmp_path / "converted.pt"
    main([fixture("gated"), "-o", str(out)])
    assert out.exists()
    assert "converted.pt" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([fixture("gated"), fixture("default"), "-o", str(out)])


def test_load_model_converts_keras_on_the_fly():
    with pytest.warns(UserWarning, match="abismal.convert"):
        model = load_model(fixture("lognormal"))
    expected, _ = load_keras(fixture("lognormal"))
    for k, v in expected.state_dict().items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_load_optimizer_from_keras():
    model, _ = load_keras(fixture("default"))
    opt = load_optimizer(fixture("default"), model)
    assert type(opt).__name__ == "Adam"
    assert opt.learning_rate == pytest.approx(1e-3)
    assert opt.state_dict()["state"] == {}


def test_not_an_abismal_file(tmp_path):
    bogus = tmp_path / "model.pt"
    torch.save({"hello": 1}, bogus)
    with pytest.raises(ValueError, match="abismal.convert"):
        load_model(str(bogus))


def test_renamed_keras_file_is_recognized(tmp_path):
    renamed = tmp_path / "model.pt"
    renamed.write_bytes(open(fixture("default"), "rb").read())
    with pytest.raises(ValueError, match="abismal 1.x"):
        load_model(str(renamed))


def test_unbuilt_model_needs_allow_missing():
    """1.x saved epoch_0.keras before the scale model was built."""
    with pytest.raises(ConversionError, match="allow.missing"):
        load_keras(fixture("unbuilt"))
    with pytest.raises(ConversionError, match="metadata"):
        load_keras(fixture("unbuilt"), allow_missing=True)
    with pytest.warns(UserWarning, match="freshly initialized"):
        model, _ = load_keras(fixture("unbuilt"), allow_missing=True, metadata_length=8)
    _, weights = read_keras(fixture("unbuilt"))
    raw = weights["surrogate_posterior/vars/0"]
    np.testing.assert_array_equal(model.surrogate_posterior.loc.raw.detach().numpy(), raw)


def _rewrite(src, dst, edit_config=None, edit_weights=None):
    """Copy a .keras file, editing its config and/or its weights."""
    import h5py
    with zipfile.ZipFile(src) as z:
        config = json.loads(z.read("config.json"))
        h5 = z.read("model.weights.h5")
    if edit_config is not None:
        edit_config(config)
    h5path = str(dst) + ".h5"
    with open(h5path, "wb") as f:
        f.write(h5)
    if edit_weights is not None:
        with h5py.File(h5path, "r+") as f:
            edit_weights(f)
    with zipfile.ZipFile(dst, "w") as z:
        z.writestr("config.json", json.dumps(config))
        z.write(h5path, "model.weights.h5")
    return str(dst)


def test_unknown_class_fails_loudly(tmp_path):
    def edit(config):
        config["config"]["surrogate_posterior"]["class_name"] = "MysteryPosterior"
    path = _rewrite(fixture("default"), tmp_path / "x.keras", edit_config=edit)
    with pytest.raises(ConversionError, match="MysteryPosterior"):
        load_keras(path)


def test_unknown_config_key_fails_loudly(tmp_path):
    def edit(config):
        config["config"]["scale_model"]["config"]["mystery_option"] = 3
    path = _rewrite(fixture("default"), tmp_path / "x.keras", edit_config=edit)
    with pytest.raises(ConversionError, match="mystery_option"):
        load_keras(path)


def test_unmapped_weight_fails_loudly(tmp_path):
    def edit(f):
        f["scale_model/layers/dense/vars"].create_dataset("1", data=np.zeros(4, dtype="float32"))
    path = _rewrite(fixture("default"), tmp_path / "x.keras", edit_weights=edit)
    with pytest.raises(ConversionError, match="dense/vars/1"):
        load_keras(path)


def test_wrong_shape_fails_loudly(tmp_path):
    def edit(f):
        del f["scale_model/output_dense/vars/1"]
        f["scale_model/output_dense/vars"].create_dataset("1", data=np.zeros(3, dtype="float32"))
    path = _rewrite(fixture("default"), tmp_path / "x.keras", edit_weights=edit)
    with pytest.raises(ConversionError, match="shape"):
        load_keras(path)


def test_not_a_zip(tmp_path):
    path = tmp_path / "x.keras"
    path.write_bytes(b"not a zip")
    with pytest.raises(ConversionError, match="not a .keras file"):
        load_keras(str(path))
