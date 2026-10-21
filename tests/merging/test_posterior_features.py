"""The scale model's pooled image representation can read the current posterior
mean and standard deviation of each reflection, and does by default."""
import pytest
import torch

from model_helpers import build_cli_model, synthetic_batch


def _batch(model):
    return synthetic_batch(model).map(torch.as_tensor)


def test_on_by_default_and_off_with_the_flag():
    on, _ = build_cli_model()
    off, _ = build_cli_model("--no-posterior-features")
    assert on.scale_model.posterior_features == 2
    assert off.scale_model.posterior_features == 0
    width = lambda m: m.scale_model.input_image.weight.shape[1]
    assert width(on) == width(off) + 2


@pytest.mark.parametrize("flags", [
    (),
    ("--prior-distribution", "normal", "--posterior-distribution", "normal", "--posterior-rank", "2"),
])
def test_the_features_are_the_posterior_mean_and_stddev(flags):
    model, _ = build_cli_model(*flags)
    batch = _batch(model)
    q = model.surrogate_posterior
    features = model._posterior_features(batch.asu_id, batch.hkl, torch.float32)
    mean = q.mean(batch.asu_id, batch.hkl).reshape(-1)
    stddev = q.stddev(batch.asu_id, batch.hkl).reshape(-1)
    assert torch.allclose(features, torch.stack([mean, stddev], -1))


@pytest.mark.parametrize("flags", [
    (),
    ("--prior-distribution", "normal", "--posterior-distribution", "normal", "--posterior-rank", "2"),
])
def test_they_carry_no_gradient_to_the_posterior(flags):
    """The scale model reads the estimates; only the likelihood and KL move them."""
    model, _ = build_cli_model(*flags)
    batch = _batch(model)
    features = model._posterior_features(batch.asu_id, batch.hkl, torch.float32)
    assert not features.requires_grad
    model.train()
    model(batch, training=True).sum().backward()
    assert all(p.grad is not None for p in model.scale_model.input_image.parameters())


def test_reflections_outside_the_asu_get_zeros():
    model, _ = build_cli_model()
    batch = _batch(model)
    hkl = batch.hkl.clone()
    hkl[0] = torch.tensor([1000, 1000, 1000])
    features = model._posterior_features(batch.asu_id, hkl, torch.float32)
    assert torch.equal(features[0], torch.zeros(2))
    assert (features[1:, 0] > 0).all()
