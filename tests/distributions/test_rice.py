"""Tests for the Rice distribution.

Both normal components used to be drawn from the same stateless seed, which made
the second exactly the first minus nu. The samples were then a deterministic
function of a single normal draw and not Rice distributed, though their mean was
close enough that nothing downstream looked wrong.
"""
import numpy as np
import pytest
import torch
from scipy.special import i0e, i1e
from scipy.stats import kstest
from scipy.stats import rice as scipy_rice

from abismal.distributions import Rice
from abismal.distributions.rice import rice_sample_gradients


# scipy parameterizes as rice(b=nu/sigma, scale=sigma)
PARAMS = [
    (0.0, 1.0),
    (0.3, 1.0),
    (2.0, 1.5),
    (5.0, 1.0),
    (30.0, 1.0),
    (39.0, 1.0),
    (41.0, 1.0),
    (300.0, 0.5),
    (1e3, 1e-3),
    (1e-3, 1e3),
]


def _mean64(nu, sigma):
    """The Rice mean in float64, sigma sqrt(pi/2) L_{1/2}(-nu^2 / 2 sigma^2).

    scipy.stats.rice.mean/var return NaN from SNR ~ 38 up."""
    x = -0.5 * (nu / sigma) ** 2
    return sigma * np.sqrt(np.pi / 2) * ((1 - x) * i0e(-x / 2) - x * i1e(-x / 2))


def _var64(nu, sigma):
    return 2 * sigma**2 + nu**2 - _mean64(nu, sigma) ** 2


@pytest.mark.parametrize("nu,sigma", [(0.0, 1.0), (0.3, 1.0), (2.0, 1.5), (5.0, 1.0), (30.0, 1.0)])
def test_reference_moments_match_scipy(nu, sigma):
    ref = scipy_rice(nu / sigma, scale=sigma)
    assert _mean64(nu, sigma) == pytest.approx(ref.mean(), rel=1e-10)
    assert _var64(nu, sigma) == pytest.approx(ref.var(), rel=1e-8)


def _dist(nu, sigma, requires_grad=False):
    nu = torch.tensor(nu, dtype=torch.float32, requires_grad=requires_grad)
    sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=requires_grad)
    return Rice(nu, sigma)


@pytest.mark.parametrize("nu,sigma", [(0.0, 1.0), (2.0, 1.5), (5.0, 1.0)])
def test_samples_match_scipy(nu, sigma):
    torch.manual_seed(1)
    z = _dist(nu, sigma).sample((20_000,)).double().numpy()
    result = kstest(z, scipy_rice(nu / sigma, scale=sigma).cdf)
    assert result.pvalue > 1e-3


def test_components_are_independent():
    torch.manual_seed(3)
    nu, sigma = 3.0, 1.0
    z = _dist(nu, sigma).sample((20_000,)).double().numpy()
    # E[z^2] = nu^2 + 2 sigma^2 for a Rice variable. With the shared seed it was
    # E[x^2 + (x - nu)^2] for x ~ N(nu, sigma), i.e. nu^2 + 2 sigma^2 as well,
    # so the second moment alone cannot tell them apart -- the variance can.
    expected_var = scipy_rice(nu / sigma, scale=sigma).var()
    assert np.isclose(z.var(), expected_var, rtol=0.05)


@pytest.mark.parametrize("nu,sigma", PARAMS)
def test_log_prob_matches_scipy(nu, sigma):
    b = nu / sigma
    lo = max(0.0, b - 5.0) * sigma
    x = np.linspace(lo, (b + 5.0) * sigma, 41)
    x = x[x > 0].astype(np.float32)
    result = _dist(nu, sigma).log_prob(torch.from_numpy(x)).numpy()
    expected = scipy_rice.logpdf(x.astype(np.float64), b, scale=sigma)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_log_prob_outside_support():
    lp = _dist(1.0, 1.0).log_prob(torch.tensor([-1.0, 0.0]))
    assert torch.all(lp == -np.inf)


@pytest.mark.parametrize("nu,sigma", PARAMS)
def test_moments_match_scipy(nu, sigma):
    dist = _dist(nu, sigma)
    b = nu / sigma
    if b > 40.0:
        # Normal crossover, as in the TF implementation.
        expected_mean, expected_var = nu, sigma**2
    else:
        expected_mean, expected_var = _mean64(nu, sigma), _var64(nu, sigma)
    assert dist.mean.item() == pytest.approx(expected_mean, rel=1e-5)
    # The variance cancels as the SNR approaches the crossover.
    rel = 1e-4 if b < 10 else 5e-3
    assert dist.variance.item() == pytest.approx(expected_var, rel=rel)
    assert dist.stddev.item() == pytest.approx(np.sqrt(expected_var), rel=rel)


def test_sample_gradients_match_the_cdf():
    """dz/dtheta = -(dF/dtheta) / f(z), with dF/dtheta by finite differences."""
    nu, sigma = 1.7, 0.9
    z = np.linspace(0.05, 5.0, 40)
    dnu, dsigma = rice_sample_gradients(
        torch.tensor(z), torch.tensor(nu, dtype=torch.float64), torch.tensor(sigma, dtype=torch.float64)
    )
    h = 1e-6
    F = lambda n, s: scipy_rice.cdf(z, n / s, scale=s)
    f = scipy_rice.pdf(z, nu / sigma, scale=sigma)
    expected_dnu = -(F(nu + h, sigma) - F(nu - h, sigma)) / (2 * h) / f
    expected_dsigma = -(F(nu, sigma + h) - F(nu, sigma - h)) / (2 * h) / f
    np.testing.assert_allclose(dnu.numpy(), expected_dnu, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(dsigma.numpy(), expected_dsigma, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("nu,sigma", [(0.0, 1.0), (0.5, 1.0), (3.0, 0.7), (60.0, 1.0)])
def test_reparameterized_gradient_of_the_mean(nu, sigma):
    """d/dtheta E[z] from samples against finite differences of scipy's mean."""
    torch.manual_seed(2)
    dist = _dist(nu, sigma, requires_grad=True)
    dist.rsample((400000,)).mean().backward()
    h = 1e-5
    mean = _mean64
    expected_dnu = (mean(nu + h, sigma) - mean(nu - h, sigma)) / (2 * h)
    expected_dsigma = (mean(nu, sigma + h) - mean(nu, sigma - h)) / (2 * h)
    assert dist.nu.grad.item() == pytest.approx(expected_dnu, abs=0.01)
    assert dist.sigma.grad.item() == pytest.approx(expected_dsigma, abs=0.01)


@pytest.mark.parametrize("nu,sigma", [(0.0, 1.0), (0.5, 1.0), (3.0, 0.7)])
def test_reparameterized_gradient_of_the_second_moment(nu, sigma):
    """E[z^2] = nu^2 + 2 sigma^2."""
    torch.manual_seed(4)
    dist = _dist(nu, sigma, requires_grad=True)
    z = dist.rsample((400000,))
    (z * z).mean().backward()
    assert dist.nu.grad.item() == pytest.approx(2 * nu, abs=0.02)
    assert dist.sigma.grad.item() == pytest.approx(4 * sigma, rel=0.01)


def test_sample_gradients_at_high_snr():
    """At x = nu z / sigma^2 ~ 1e12, torch's i1e/i0e rounds above 1, which put
    errors of ~0.1 into dz/dsigma; the asymptotic series does not."""
    z = torch.tensor([1e3, 1e3 + 1e-3 * 0.5])
    dnu, dsigma = rice_sample_gradients(z, torch.tensor(1e3), torch.tensor(1e-3))
    assert torch.all(dnu <= 1.0)
    # dz/dsigma = (z - nu)/sigma + sigma^2/(2 z sigma) + O(sigma^4)
    expected = (z.double() - 1e3) / 1e-3 + 1e-3 / (2 * z.double())
    np.testing.assert_allclose(dsigma.double().numpy(), expected.numpy(), atol=1e-6)


@pytest.mark.parametrize("x", [1.0, 29.0, 31.0, 1e3, 1e9])
def test_bessel_ratio_matches_scipy(x):
    from abismal.distributions.rice import _one_minus_bessel_ratio
    expected = 1.0 - i1e(x) / i0e(x)
    result = _one_minus_bessel_ratio(torch.tensor(x, dtype=torch.float32)).item()
    # Below the series threshold 1 - i1e/i0e cancels to ~2 x eps relative; the
    # error that leaves in dz/dsigma is that times sigma / 2z, far below eps.
    assert result == pytest.approx(expected, rel=5e-6)
