import numpy as np
import pytest
import torch
from scipy.special import gammaln
from scipy.stats import kstest
from scipy.stats import nakagami as scipy_nakagami
from torch.distributions import Gamma, HalfNormal, Weibull, kl_divergence

from abismal.distributions import Nakagami, log_gamma_half_ratio, log_gamma_half_ratio_excess

PARAMS = [
    (0.5, 1.0),
    (1.0, 1.5),
    (2.5, 0.7),
    (4.0, 3.1),
    (37.0, 2.0),
    (1e3, 1.0),
    (1e5, 4.0),
]


@pytest.mark.parametrize("m,omega", PARAMS[:4])
def test_sample_shape_and_support(m, omega):
    torch.manual_seed(0)
    samples = Nakagami(m, omega).sample((1000,))
    assert samples.shape == (1000,)
    assert torch.isfinite(samples).all()
    assert (samples >= 0.0).all()


def test_sample_shape_broadcasting():
    m = torch.tensor([0.5, 1.0, 1.5])
    omega = torch.tensor([[1.0], [2.0]])
    samples = Nakagami(m, omega).sample((50,))
    assert samples.shape == (50, 2, 3)


@pytest.mark.parametrize("m,omega", PARAMS[:4])
def test_samples_match_scipy(m, omega):
    torch.manual_seed(1)
    z = Nakagami(m, omega).sample((20000,)).double().numpy()
    assert kstest(z, scipy_nakagami(m, scale=np.sqrt(omega)).cdf).pvalue > 1e-3


@pytest.mark.parametrize("m,omega", PARAMS)
def test_log_prob_matches_scipy(m, omega):
    d = scipy_nakagami(m, scale=np.sqrt(omega))
    x = np.linspace(d.ppf(1e-4), d.ppf(1 - 1e-4), 50).astype(np.float32)
    log_prob = Nakagami(m, omega).log_prob(torch.from_numpy(x)).numpy()
    expected = d.logpdf(x.astype(np.float64))
    # float32 resolves log1p(u) - u, u ~ 1/sqrt(m), to ~eps |u|, which the
    # density multiplies by m: ~5e-5 absolute at m = 1e5.
    atol = 1e-4
    np.testing.assert_allclose(log_prob, expected, rtol=1e-4, atol=atol)


def test_cdf_matches_scipy():
    x = torch.linspace(0.0, 4.0, 30)
    np.testing.assert_allclose(
        Nakagami(1.7, 2.1).cdf(x).numpy(),
        scipy_nakagami.cdf(x.numpy(), 1.7, scale=np.sqrt(2.1)),
        atol=1e-6,
    )


def test_centric_wilson_matches_half_normal():
    sigma = 1.7
    x = torch.linspace(0.01, 5.0, 40)
    np.testing.assert_allclose(
        Nakagami(0.5, sigma * sigma).log_prob(x).numpy(),
        HalfNormal(sigma).log_prob(x).numpy(),
        rtol=1e-4, atol=1e-5)


def test_acentric_wilson_matches_weibull():
    sigma = 1.7
    x = torch.linspace(0.01, 5.0, 40)
    np.testing.assert_allclose(
        Nakagami(1.0, sigma * sigma).log_prob(x).numpy(),
        Weibull(sigma, 2.0).log_prob(x).numpy(),
        rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("m,omega", [(0.7, 1.0), (2.0, 1.5), (10.0, 0.5)])
def test_reparameterized_gradients(m, omega):
    """d/dtheta E[y] from samples against the analytic mean's gradient."""
    torch.manual_seed(1)
    mt = torch.tensor(m, requires_grad=True)
    ot = torch.tensor(omega, requires_grad=True)
    Nakagami(mt, ot).rsample((400000,)).mean().backward()
    sample_grads = [mt.grad.item(), ot.grad.item()]

    h = 1e-5
    mean = lambda m, o: scipy_nakagami.mean(m, scale=np.sqrt(o))
    expected = [
        (mean(m + h, omega) - mean(m - h, omega)) / (2 * h),
        (mean(m, omega + h) - mean(m, omega - h)) / (2 * h),
    ]
    np.testing.assert_allclose(sample_grads, expected, atol=5e-3)


def test_kl_self_is_zero():
    p = Nakagami(2.0, 1.3)
    np.testing.assert_allclose(kl_divergence(p, p).item(), 0.0, atol=1e-6)


def test_kl_matches_gamma_kl():
    m1, o1 = 1.5, 1.2
    m2, o2 = 2.3, 0.8
    kl_nak = kl_divergence(Nakagami(m1, o1), Nakagami(m2, o2)).item()
    kl_gam = kl_divergence(Gamma(m1, m1 / o1), Gamma(m2, m2 / o2)).item()
    np.testing.assert_allclose(kl_nak, kl_gam, rtol=1e-6, atol=1e-7)


def test_kl_matches_monte_carlo():
    torch.manual_seed(42)
    p = Nakagami(2.5, 1.2)
    q = Nakagami(1.7, 0.9)
    kl_analytic = kl_divergence(p, q).item()
    samples = p.sample((200_000,))
    mc = (p.log_prob(samples) - q.log_prob(samples)).mean().item()
    assert abs(kl_analytic - mc) < 0.01


def test_kl_is_registered_analytically():
    result = kl_divergence(Nakagami(1.5, 1.0), Nakagami(2.0, 1.5))
    assert torch.isfinite(result)
    assert result.item() >= 0.0


@pytest.mark.parametrize("m,omega", PARAMS)
def test_mean_matches_scipy(m, omega):
    expected = scipy_nakagami.mean(m, scale=np.sqrt(omega))
    np.testing.assert_allclose(Nakagami(m, omega).mean.item(), expected, rtol=1e-5)


@pytest.mark.parametrize("m,omega", PARAMS)
def test_variance_and_stddev_match_scipy(m, omega):
    # Omega (1 - Gamma(m + 1/2)^2 / (m Gamma(m)^2)) in float64 from the excess
    # r = lgamma(m + 1/2) - lgamma(m) - log(m)/2; scipy's own var() cancels for
    # m >~ 1e4.
    r = gammaln(m + 0.5) - gammaln(m) - 0.5 * np.log(m)
    if m > 100:
        r = -1 / (8 * m) + 1 / (192 * m**3)
    expected_var = -omega * np.expm1(2 * r)
    d = Nakagami(m, omega)
    np.testing.assert_allclose(d.variance.item(), expected_var, rtol=1e-4)
    np.testing.assert_allclose(d.stddev.item(), np.sqrt(expected_var), rtol=1e-4)
    if m < 100:
        np.testing.assert_allclose(
            expected_var, scipy_nakagami.var(m, scale=np.sqrt(omega)), rtol=1e-6)


@pytest.mark.parametrize("m,omega", PARAMS)
def test_mode_matches_closed_form(m, omega):
    expected = np.sqrt(max(0.0, (2.0 * m - 1.0) * omega / (2.0 * m)))
    np.testing.assert_allclose(Nakagami(m, omega).mode.item(), expected, rtol=1e-5, atol=1e-6)


def test_square_returns_gamma_with_matched_parameters():
    m, omega = 2.3, 1.7
    sq = Nakagami(m, omega).square()
    assert isinstance(sq, Gamma)
    np.testing.assert_allclose(sq.concentration.item(), m, rtol=1e-6)
    np.testing.assert_allclose(sq.rate.item(), m / omega, rtol=1e-6)


def test_square_distribution_matches_monte_carlo():
    torch.manual_seed(7)
    d = Nakagami(1.8, 2.1)
    sq = d.square()
    y = d.sample((200_000,))
    ysq = y * y
    np.testing.assert_allclose(ysq.mean().item(), sq.mean.item(), rtol=1e-2)
    np.testing.assert_allclose(ysq.var().item(), sq.variance.item(), rtol=5e-2)


def test_log_gamma_half_ratio_matches_scipy():
    m = np.geomspace(0.5, 1e6, 300)
    expected = gammaln(m + 0.5) - gammaln(m)
    result = log_gamma_half_ratio(torch.tensor(m, dtype=torch.float32)).double().numpy()
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-7)


def test_log_gamma_half_ratio_excess_matches_scipy():
    # Compared where float64 gammaln still resolves the excess (~ -1/(8m)).
    m = np.geomspace(0.5, 1e3, 200)
    expected = gammaln(m + 0.5) - gammaln(m) - 0.5 * np.log(m)
    result = log_gamma_half_ratio_excess(torch.tensor(m, dtype=torch.float32)).double().numpy()
    np.testing.assert_allclose(result, expected, rtol=5e-6)
