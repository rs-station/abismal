import numpy as np
import pytest
import torch
from scipy.stats import kstest
from scipy.stats import truncnorm as scipy_truncnorm

from abismal.distributions import TruncatedNormal

# (loc, scale, low, high). The first rows are shaped like the structure factor
# posterior, TruncatedNormal(loc + eps, scale, eps, 1e10).
PARAMS = [
    (1.0, 1.0, 1e-12, 1e10),
    (5.0, 0.01, 1e-12, 1e10),
    (0.01, 1.0, 1e-12, 1e10),
    (1e3, 1e-3, 1e-12, 1e10),
    (1e-3, 1e3, 1e-12, 1e10),
    (0.0, 1.0, -1.0, 1.0),
    (0.0, 1.0, 3.0, 5.0),
    (0.0, 1.0, -5.0, -3.0),
    (0.0, 1.0, 20.0, 25.0),
    (0.0, 1.0, -1.0, np.inf),
    (2.0, 0.5, -np.inf, 1.5),
]


def _ref(loc, scale, low, high):
    return scipy_truncnorm((low - loc) / scale, (high - loc) / scale, loc=loc, scale=scale)


def _dist(loc, scale, low, high, requires_grad=False):
    loc = torch.tensor(loc, dtype=torch.float32, requires_grad=requires_grad)
    scale = torch.tensor(scale, dtype=torch.float32, requires_grad=requires_grad)
    return TruncatedNormal(loc, scale, low, high)


@pytest.mark.parametrize("loc,scale,low,high", PARAMS)
def test_log_prob_matches_scipy(loc, scale, low, high):
    ref = _ref(loc, scale, low, high)
    x = np.linspace(ref.ppf(1e-3), ref.ppf(1 - 1e-3), 30).astype(np.float32)
    x = np.clip(x, np.float32(low), np.float32(high))
    result = _dist(loc, scale, low, high).log_prob(torch.from_numpy(x)).numpy()
    expected = ref.logpdf(x.astype(np.float64))
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)


def test_log_prob_outside_support():
    lp = _dist(0.0, 1.0, -1.0, 1.0).log_prob(torch.tensor([-1.5, 1.5]))
    assert torch.all(lp == -np.inf)


@pytest.mark.parametrize("loc,scale,low,high", PARAMS)
def test_moments_match_scipy(loc, scale, low, high):
    ref = _ref(loc, scale, low, high)
    d = _dist(loc, scale, low, high)
    assert d.mean.item() == pytest.approx(ref.mean(), rel=1e-5, abs=1e-5 * scale)
    # The variance formula cancels deep in a tail (documented); |a| <= 20 here.
    rel = 1e-2 if low >= 20.0 else 1e-4
    assert d.variance.item() == pytest.approx(ref.var(), rel=rel)


@pytest.mark.parametrize("loc,scale,low,high", PARAMS)
def test_samples_match_scipy(loc, scale, low, high):
    torch.manual_seed(0)
    z = _dist(loc, scale, low, high).sample((20000,))
    assert torch.all((z >= low) & (z <= high))
    if scale / max(abs(loc), 1.0) < 1e-4:
        # float32 cannot resolve the spread of these samples around loc.
        return
    assert kstest(z.double().numpy(), _ref(loc, scale, low, high).cdf).pvalue > 1e-3


@pytest.mark.parametrize("loc,scale,low,high", [
    (1.0, 1.0, 1e-12, 1e10),
    (0.3, 1.0, 1e-12, 1e10),
    (0.0, 1.0, -1.0, 2.0),
    (0.0, 1.0, 3.0, 5.0),
])
def test_reparameterized_gradient_of_the_mean(loc, scale, low, high):
    """d/dtheta E[z] from samples against finite differences of scipy's mean."""
    torch.manual_seed(1)
    d = _dist(loc, scale, low, high, requires_grad=True)
    d.rsample((400000,)).mean().backward()
    h = 1e-5
    mean = lambda l, s: _ref(l, s, low, high).mean()
    expected_dloc = (mean(loc + h, scale) - mean(loc - h, scale)) / (2 * h)
    expected_dscale = (mean(loc, scale + h) - mean(loc, scale - h)) / (2 * h)
    assert d.loc.grad.item() == pytest.approx(expected_dloc, abs=0.01)
    assert d.scale.grad.item() == pytest.approx(expected_dscale, abs=0.01)
