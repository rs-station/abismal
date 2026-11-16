"""--scale-prior-mode decoded: each observation's LogNormal scale prior is decoded from the leave-one-out
pooled image representation, under a LogNormal(0, 1) hyperprior counted once per observation."""
import pytest
import torch
import torch.distributions as td

from model_helpers import build_cli_model, synthetic_batch
from abismal import tracking
from abismal.scaling.scaling import kl_divergence


def _batch(model):
    return synthetic_batch(model).map(torch.as_tensor)


def _q(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    loc = torch.rand(n, generator=g) + 0.5
    scale = torch.rand(n, generator=g) * 0.3 + 0.1
    q = td.LogNormal(loc.log(), scale)
    return q, q.sample((8,))


def test_flags():
    fixed, _ = build_cli_model()
    decoded, _ = build_cli_model("--scale-prior-mode", "decoded")
    attached, _ = build_cli_model("--scale-prior-mode", "decoded", "--no-scale-prior-detach")
    assert fixed.scale_model.prior_mode == 'fixed' and not hasattr(fixed.scale_model, 'prior_dense')
    assert decoded.scale_model.prior_mode == 'decoded' and decoded.scale_model.prior_detach
    assert not attached.scale_model.prior_detach


def test_other_weights_initialize_as_without_it():
    fixed, _ = build_cli_model()
    decoded, _ = build_cli_model("--scale-prior-mode", "decoded")
    d = decoded.scale_model.state_dict()
    for k, v in fixed.scale_model.state_dict().items():
        assert torch.equal(v, d[k]), k


def test_it_starts_at_the_fixed_lognormal_prior():
    model, _ = build_cli_model("--scale-prior-mode", "decoded")
    sm = model.scale_model
    n = 12
    pooled = torch.randn(n, sm.mlp_width)
    mu, sigma = sm.decoded_prior(pooled)
    assert torch.equal(mu, torch.zeros(n)) and torch.equal(sigma, torch.ones(n))
    q, z = _q(n)
    got = sm._decoded_prior_kl(q, z, pooled)
    want = kl_divergence(q, td.LogNormal(0., 1.), z, sm.epsilon).mean()
    assert torch.allclose(got, want)


def test_the_hyperprior_counts_once_per_observation():
    model, _ = build_cli_model("--scale-prior-mode", "decoded")
    sm = model.scale_model
    n = 10
    mus, sigmas = torch.linspace(-1., 2., n), torch.linspace(0.5, 2., n)
    sm.decoded_prior = lambda pooled: (mus, sigmas)
    q, z = _q(n)
    got = sm._decoded_prior_kl(q, z, None)
    kl_obs = kl_divergence(q, td.LogNormal(mus, sigmas), z, sm.epsilon).mean()
    hyper = td.kl_divergence(td.Normal(mus, sigmas), td.Normal(0., 1.)).mean()
    assert torch.allclose(got, kl_obs + hyper)


@pytest.mark.parametrize("detach", [True, False])
def test_detach_stops_the_gradient_into_the_pooled_representation(detach):
    flags = ("--scale-prior-mode", "decoded") + (() if detach else ("--no-scale-prior-detach",))
    model, _ = build_cli_model(*flags)
    sm = model.scale_model
    with torch.no_grad():  # move the decoder off its zero start so the gradient can flow
        sm.prior_dense.weight.normal_()
    n = 6
    pooled = torch.randn(n, sm.mlp_width, requires_grad=True)
    q, z = _q(n)
    sm._decoded_prior_kl(q, z, pooled).backward()
    assert sm.prior_dense.weight.grad is not None
    if detach:
        assert pooled.grad is None
    else:
        assert pooled.grad is not None and pooled.grad.abs().sum() > 0


def test_it_trains():
    model, _ = build_cli_model("--scale-prior-mode", "decoded")
    batch = _batch(model)
    model.train()
    with tracking.track() as t:
        model(batch, training=True)
    assert 'KL_Σhyper' in t.metrics and 'Σprior_μ' in t.metrics
    t.total_loss().backward()
    assert all(p.grad is not None for p in model.scale_model.prior_dense.parameters())
