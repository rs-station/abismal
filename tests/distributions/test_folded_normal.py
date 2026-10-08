"""Tests for the folded normal distribution.

The CDF is checked against scipy because it went wrong quietly: it read
`erf(a) - erf(b)` where the folded normal CDF is the sum, and nothing in abismal
calls cdf(), so no result ever disagreed with anything. The sampler's
implicit-reparameterization gradients are derived from the *correct* CDF, so the
two were inconsistent with each other -- which is what these tests pin down.
"""
import numpy as np
import pytest
import torch
from scipy.stats import foldnorm as scipy_foldnorm
from scipy.stats import kstest

from abismal.distributions import FoldedNormal
from abismal.distributions.folded_normal import folded_normal_sample_gradients


# scipy parameterizes as foldnorm(c=loc/scale, scale=scale)
PARAMS = [
    (0.0, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (3.0, 0.5),
    (0.2, 2.5),
]

# Wider grid for the moments, including both sides of the |loc/scale| = 10
# crossover to the normal and very large ratios.
MOMENT_PARAMS = PARAMS + [
    (-2.0, 1.0),
    (9.0, 1.0),
    (11.0, 1.0),
    (100.0, 0.5),
    (1e3, 1e-3),
    (1e-3, 1e3),
    (-50.0, 2.0),
]


def _dist(loc, scale, requires_grad=False):
    loc = torch.tensor(loc, dtype=torch.float32, requires_grad=requires_grad)
    scale = torch.tensor(scale, dtype=torch.float32, requires_grad=requires_grad)
    return FoldedNormal(loc, scale)


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_matches_scipy(loc, scale):
    x = torch.linspace(0.0, loc + 6.0 * scale, 50)
    result = _dist(loc, scale).cdf(x).numpy()
    expected = scipy_foldnorm.cdf(x.numpy() / scale, loc / scale)
    assert np.allclose(result, expected, atol=1e-5), (
        f"max deviation {np.abs(result - expected).max():.4g}"
    )


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_is_a_distribution_function(loc, scale):
    """Non-decreasing, in [0, 1], and zero below the support."""
    x = torch.linspace(-3.0 * scale, loc + 8.0 * scale, 200)
    cdf = _dist(loc, scale).cdf(x).numpy()
    x = x.numpy()

    assert np.all(cdf >= 0.0), f"negative CDF, min {cdf.min():.4g}"
    assert np.all(cdf <= 1.0 + 1e-6), f"CDF above 1, max {cdf.max():.4g}"
    assert np.all(np.diff(cdf) >= -1e-6), "CDF is not non-decreasing"
    assert np.all(cdf[x < 0] == 0.0), "CDF is non-zero below the support"


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_agrees_with_the_sampler(loc, scale):
    """The empirical CDF of the samples must follow cdf().

    cdf() and rsample are independent code paths, and it was their
    disagreement that the sign error amounted to.
    """
    torch.manual_seed(0)
    dist = _dist(loc, scale)
    samples = dist.sample((200000,)).numpy()
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        x = np.quantile(samples, q)
        assert dist.cdf(torch.tensor(x)).item() == pytest.approx(q, abs=0.01)


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_is_the_integral_of_the_density(loc, scale):
    """cdf() against a numerical integral of exp(log_prob), the other direction."""
    dist = _dist(loc, scale)
    x = torch.linspace(0.0, loc + 8.0 * scale, 20001)
    pdf = dist.log_prob(x).exp().numpy()
    xn = x.numpy()
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xn))])
    assert np.allclose(dist.cdf(x).numpy(), integral, atol=2e-4)


@pytest.mark.parametrize("loc,scale", MOMENT_PARAMS)
def test_log_prob_matches_scipy(loc, scale):
    c = abs(loc) / scale
    x = np.linspace(max(0.0, (c - 5.0) * scale), (c + 5.0) * scale, 41)
    x = x[x > 0].astype(np.float32)
    result = _dist(loc, scale).log_prob(torch.from_numpy(x)).numpy()
    expected = scipy_foldnorm.logpdf(x.astype(np.float64), c, scale=scale)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("loc,scale", MOMENT_PARAMS)
def test_moments_match_scipy(loc, scale):
    dist = _dist(loc, scale)
    c = abs(loc) / scale
    expected_mean = scipy_foldnorm.mean(c, scale=scale)
    expected_var = scale**2 if c >= 10 else scipy_foldnorm.var(c, scale=scale)
    assert dist.mean.item() == pytest.approx(expected_mean, rel=1e-5, abs=1e-6 * scale)
    assert dist.variance.item() == pytest.approx(expected_var, rel=1e-4)
    assert dist.stddev.item() == pytest.approx(np.sqrt(expected_var), rel=1e-4)


@pytest.mark.parametrize("loc,scale", PARAMS + [(20.0, 1.0)])
def test_samples_match_scipy(loc, scale):
    torch.manual_seed(1)
    z = _dist(loc, scale).sample((20000,)).double().numpy()
    result = kstest(z, scipy_foldnorm(abs(loc) / scale, scale=scale).cdf)
    assert result.pvalue > 1e-3


def test_sample_gradients_match_the_cdf():
    """dz/dtheta = -(dF/dtheta) / f(z), with dF/dtheta by finite differences."""
    loc, scale = 0.7, 1.3
    z = np.linspace(0.01, 5.0, 50)
    dloc, dscale = folded_normal_sample_gradients(
        torch.tensor(z), torch.tensor(loc, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64)
    )
    h = 1e-6
    F = lambda l, s: scipy_foldnorm.cdf(z, l / s, scale=s)
    f = scipy_foldnorm.pdf(z, loc / scale, scale=scale)
    expected_dloc = -(F(loc + h, scale) - F(loc - h, scale)) / (2 * h) / f
    expected_dscale = -(F(loc, scale + h) - F(loc, scale - h)) / (2 * h) / f
    np.testing.assert_allclose(dloc.numpy(), expected_dloc, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dscale.numpy(), expected_dscale, rtol=1e-5, atol=1e-7)


def test_sample_gradients_stay_finite_far_in_the_tail():
    """Both normal densities underflow here; the direct ratio was 0/0."""
    z = torch.tensor([60.0, 200.0])
    dloc, dscale = folded_normal_sample_gradients(z, torch.tensor(0.5), torch.tensor(1.0))
    assert torch.isfinite(dloc).all() and torch.isfinite(dscale).all()
    np.testing.assert_allclose(dloc.numpy(), 1.0)
    np.testing.assert_allclose(dscale.numpy(), (z.numpy() - 0.5) / 1.0, rtol=1e-6)


@pytest.mark.parametrize("loc,scale", [(0.0, 1.0), (0.5, 1.0), (2.0, 0.7), (15.0, 1.0)])
def test_reparameterized_gradient_of_the_mean(loc, scale):
    """d/dtheta E[z] from samples against finite differences of scipy's mean."""
    torch.manual_seed(2)
    dist = _dist(loc, scale, requires_grad=True)
    dist.rsample((400000,)).mean().backward()
    h = 1e-5
    mean = lambda l, s: scipy_foldnorm.mean(abs(l) / s, scale=s)
    expected_dloc = (mean(loc + h, scale) - mean(loc - h, scale)) / (2 * h)
    expected_dscale = (mean(loc, scale + h) - mean(loc, scale - h)) / (2 * h)
    assert dist.loc.grad.item() == pytest.approx(expected_dloc, abs=0.01)
    assert dist.scale.grad.item() == pytest.approx(expected_dscale, abs=0.01)


@pytest.mark.parametrize("loc,scale", [(0.0, 1.0), (0.5, 1.0), (2.0, 0.7)])
def test_reparameterized_gradient_of_the_second_moment(loc, scale):
    """E[z^2] = loc^2 + scale^2, whatever the folding."""
    torch.manual_seed(3)
    dist = _dist(loc, scale, requires_grad=True)
    z = dist.rsample((400000,))
    (z * z).mean().backward()
    assert dist.loc.grad.item() == pytest.approx(2 * loc, abs=0.02)
    assert dist.scale.grad.item() == pytest.approx(2 * scale, rel=0.01)
