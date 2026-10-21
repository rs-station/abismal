"""A checkpoint holds the arguments and data facts a model was built from and its
weights; loading rebuilds the model with abismal.build and loads the weights."""
import pytest
import torch

from abismal.saving import load_args, load_model, load_weights, save_model
from model_helpers import build_cli_model, build_small_model


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def test_round_trip(tmp_path):
    model, _ = build_cli_model("--posterior-distribution", "rice")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    save_model(model, tmp_path / "m.pt")
    reloaded = load_model(tmp_path / "m.pt")
    assert type(reloaded.surrogate_posterior).__name__ == "RicePosterior"
    assert _same_weights(model, reloaded)


def test_the_file_holds_only_plain_types(tmp_path):
    """Paths and the like are stored as strings, so torch's safe loader accepts the file."""
    model, _ = build_cli_model()
    save_model(model, tmp_path / "m.pt")
    payload = torch.load(tmp_path / "m.pt", weights_only=True)
    assert payload["args"]["out_dir"] == "unused"
    assert payload["facts"]["spacegroup"] == "P 21 21 21"


def test_load_args(tmp_path):
    model, _ = build_cli_model("--mtz-metadata", "XDET,YDET")
    save_model(model, tmp_path / "m.pt")
    args = load_args(tmp_path / "m.pt")
    assert args.inputs == ["unused.mtz"]
    assert args.mtz_metadata == "XDET,YDET"


def test_an_option_added_later_takes_its_default(tmp_path):
    model, _ = build_cli_model()
    save_model(model, tmp_path / "m.pt")
    payload = torch.load(tmp_path / "m.pt", weights_only=True)
    del payload["args"]["gated"]
    torch.save(payload, tmp_path / "m.pt")
    assert _same_weights(model, load_model(tmp_path / "m.pt"))


def test_load_weights_takes_one_component(tmp_path):
    source, _ = build_cli_model()
    with torch.no_grad():
        for p in source.parameters():
            p.add_(1.0)
    save_model(source, tmp_path / "m.pt")
    target, _ = build_cli_model()
    load_weights(tmp_path / "m.pt", target.scale_model, "scale_model")
    assert _same_weights(source.scale_model, target.scale_model)
    assert not _same_weights(source.surrogate_posterior, target.surrogate_posterior)


def test_a_checkpoint_without_arguments_cannot_be_rebuilt(tmp_path):
    model, _ = build_cli_model()
    torch.save({"format": "abismal-torch", "format_version": 1, "config": "{}",
                "state_dict": model.state_dict()}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="did not record its arguments"):
        load_model(tmp_path / "old.pt")
    # Its weights still serve as an initialization.
    load_weights(tmp_path / "old.pt", model.surrogate_posterior, "surrogate_posterior")


def test_a_hand_built_model_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="not built by abismal.build"):
        save_model(build_small_model(), tmp_path / "m.pt")


def test_a_checkpoint_from_before_hidden_multiplier_keeps_its_hidden_width(tmp_path):
    """Checkpoints saved before --hidden-multiplier existed had hidden layers twice
    --d-model wide; today's default of 2/3 must not be applied to them."""
    model, _ = build_cli_model("--hidden-multiplier", "2", "--no-gated", "--pre-activation", "relu", "--activation", "relu")
    save_model(model, tmp_path / "m.pt")
    payload = torch.load(tmp_path / "m.pt", weights_only=True)
    del payload["args"]["hidden_multiplier"]
    torch.save(payload, tmp_path / "m.pt")
    assert _same_weights(model, load_model(tmp_path / "m.pt"))


def test_the_default_model_is_finite_before_training():
    """An untrained model's standardization used to divide by a variance of
    m2/0 = inf, making every input row zero, and RMS normalization of a zero row
    is 0/0. Standardization with no data seen is now the identity."""
    from model_helpers import synthetic_batch
    model, _ = build_cli_model()
    model.eval()
    with torch.no_grad():
        out = model(synthetic_batch(model).map(torch.as_tensor), training=False)
    assert torch.isfinite(out).all()


def test_a_checkpoint_from_before_posterior_features_is_rebuilt_without_them(tmp_path):
    model, _ = build_cli_model("--no-posterior-features")
    save_model(model, tmp_path / "m.pt")
    payload = torch.load(tmp_path / "m.pt", weights_only=True)
    del payload["args"]["posterior_features"]
    torch.save(payload, tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    assert loaded.scale_model.posterior_features == 0
    assert _same_weights(model, loaded)
