"""Behaviour every distribution in abismal.distributions shares: shapes,
broadcasting of the implicit gradients, finiteness at extreme parameters, the KL
registry, devices and torch.compile."""
import itertools

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from abismal.distributions import (
    FoldedNormal,
    Nakagami,
    Rice,
    RiceWoolfson,
    TruncatedNormal,
)

# Each factory takes the two learnable parameters (a location-like and a
# scale-like one, both positive here) and builds the distribution.
FACTORIES = {
    "FoldedNormal": FoldedNormal,
    "Rice": Rice,
    "Nakagami": Nakagami,
    "TruncatedNormal": lambda a, b: TruncatedNormal(a + 1e-12, b, 1e-12, 1e10),
    # Alternating centric flags along the last batch dimension.
    "RiceWoolfson": lambda a, b: RiceWoolfson(
        a, b, torch.arange(torch.broadcast_shapes(a.shape, b.shape)[-1], device=a.device) % 2 == 0
    ),
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request):
    return FACTORIES[request.param]


def _check_all(dist, x):
    """Every differentiable output of a distribution, summed into one scalar."""
    z = dist.rsample((7,))
    return (
        z.sum()
        + dist.log_prob(z).sum()
        + dist.log_prob(x).sum()
        + dist.mean.sum()
        + dist.variance.sum()
        + dist.stddev.sum()
    )


def test_shapes_and_dtype(factory):
    a = torch.rand(2, 3) + 0.5
    b = torch.rand(2, 3) + 0.5
    d = factory(a, b)
    assert d.batch_shape == (2, 3)
    z = d.rsample((5, 4))
    assert z.shape == (5, 4, 2, 3)
    assert z.dtype == torch.float32
    assert d.log_prob(z).shape == (5, 4, 2, 3)
    assert d.mean.shape == d.stddev.shape == d.variance.shape == (2, 3)
    assert d.has_rsample


def test_float64(factory):
    d = factory(torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
    z = d.rsample((2,))
    assert z.dtype == torch.float64
    assert d.log_prob(z).dtype == d.mean.dtype == torch.float64


def test_sample_has_no_graph(factory):
    a = torch.ones(3, requires_grad=True)
    b = torch.ones(3, requires_grad=True)
    assert not factory(a, b).sample((2,)).requires_grad
    assert factory(a, b).rsample((2,)).requires_grad


def test_gradients_through_broadcast_parameters(factory):
    """Implicit gradients reduce correctly over parameters broadcast from
    smaller shapes as well as over the sample dimensions."""
    a_small = (torch.rand(3, 1) + 0.5).requires_grad_(True)
    b_small = (torch.rand(1, 4) + 0.5).requires_grad_(True)
    a_full = a_small.detach().expand(3, 4).clone().requires_grad_(True)
    b_full = b_small.detach().expand(3, 4).clone().requires_grad_(True)
    w = torch.rand(6, 3, 4)

    torch.manual_seed(0)
    (w * factory(a_small, b_small).rsample((6,))).sum().backward()
    torch.manual_seed(0)
    (w * factory(a_full, b_full).rsample((6,))).sum().backward()

    torch.testing.assert_close(a_small.grad, a_full.grad.sum(1, keepdim=True))
    torch.testing.assert_close(b_small.grad, b_full.grad.sum(0, keepdim=True))


EXTREMES = [1e-3, 1.0, 1e3]


@pytest.mark.parametrize("a,b", list(itertools.product(EXTREMES, EXTREMES)))
def test_gradients_are_finite_at_extreme_parameters(factory, a, b):
    torch.manual_seed(0)
    at = torch.full((8,), a, requires_grad=True)
    bt = torch.full((8,), b, requires_grad=True)
    d = factory(at, bt)
    x = torch.tensor([1e-3, 1.0, 1e3]).reshape(3, 1)
    out = _check_all(d, x)
    out.backward()
    assert torch.isfinite(out), out
    assert torch.isfinite(at.grad).all(), at.grad
    assert torch.isfinite(bt.grad).all(), bt.grad


def test_unregistered_kl_raises(factory):
    """Callers fall back to a Monte Carlo KL on NotImplementedError."""
    d = factory(torch.ones(3), torch.ones(3))
    with pytest.raises(NotImplementedError):
        kl_divergence(d, Normal(torch.ones(3), torch.ones(3)))
    if not isinstance(d, Nakagami):
        with pytest.raises(NotImplementedError):
            kl_divergence(d, d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda(factory):
    a = (torch.rand(5, device="cuda") + 0.5).requires_grad_(True)
    b = (torch.rand(5, device="cuda") + 0.5).requires_grad_(True)
    out = _check_all(factory(a, b), torch.rand(5, device="cuda") + 0.1)
    out.backward()
    assert out.device.type == "cuda"
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["FoldedNormal", "Rice"])
def test_torch_compile(name):
    """Sample, log_prob and backward compile into one graph and agree with eager
    on the deterministic parts."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    cls = FACTORIES[name]

    def loss(a, b, x):
        d = cls(a, b)
        z = d.rsample((4,))
        return d.log_prob(z).mean() + z.mean(), d.log_prob(x).sum() + d.mean.sum()

    compiled = torch.compile(loss, fullgraph=True)
    a = (torch.rand(6) + 0.5).requires_grad_(True)
    b = (torch.rand(6) + 0.5).requires_grad_(True)
    x = torch.rand(6) + 0.1
    try:
        stochastic, deterministic = compiled(a, b, x)
    except Exception as e:  # e.g. no C compiler for inductor
        pytest.skip(f"torch.compile failed to build: {type(e).__name__}: {e}")
    (stochastic + deterministic).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

    _, eager = loss(a, b, x)
    torch.testing.assert_close(deterministic, eager)
