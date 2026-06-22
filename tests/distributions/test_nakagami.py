import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import nakagami as scipy_nakagami

from abismal.distributions import Nakagami

tfd = tfp.distributions


@pytest.mark.parametrize("m,omega", [
    (0.5, 1.0),
    (1.0, 1.0),
    (1.5, 2.3),
    (3.0, 0.7),
])
def test_sample_shape_and_support(m, omega):
    dist = Nakagami(m, omega)
    samples = dist.sample(1000, seed=0)
    assert samples.shape == (1000,)
    assert tf.reduce_all(tf.math.is_finite(samples)).numpy()
    assert tf.reduce_all(samples >= 0.0).numpy()


def test_sample_shape_broadcasting():
    m = tf.constant([0.5, 1.0, 1.5])
    omega = tf.constant([[1.0], [2.0]])
    dist = Nakagami(m, omega)
    samples = dist.sample(50, seed=0)
    assert samples.shape == (50, 2, 3)


@pytest.mark.parametrize("m,omega", [
    (0.5, 1.0),
    (1.0, 1.5),
    (2.5, 0.7),
    (4.0, 3.1),
])
def test_log_prob_matches_scipy(m, omega):
    dist = Nakagami(m, omega)
    x = np.linspace(0.05, 5.0, 50).astype(np.float32)
    log_prob = dist.log_prob(x).numpy()
    expected = scipy_nakagami.logpdf(x, m, scale=np.sqrt(omega))
    np.testing.assert_allclose(log_prob, expected, rtol=1e-4, atol=1e-5)


def test_centric_wilson_matches_half_normal():
    sigma = 1.7
    dist = Nakagami(0.5, sigma * sigma)
    reference = tfd.HalfNormal(scale=sigma)
    x = np.linspace(0.01, 5.0, 40).astype(np.float32)
    np.testing.assert_allclose(
        dist.log_prob(x).numpy(), reference.log_prob(x).numpy(),
        rtol=1e-4, atol=1e-5)


def test_acentric_wilson_matches_weibull():
    sigma = 1.7
    dist = Nakagami(1.0, sigma * sigma)
    reference = tfd.Weibull(concentration=2.0, scale=sigma)
    x = np.linspace(0.01, 5.0, 40).astype(np.float32)
    np.testing.assert_allclose(
        dist.log_prob(x).numpy(), reference.log_prob(x).numpy(),
        rtol=1e-4, atol=1e-5)


def test_reparameterized_gradients():
    m = tf.Variable(2.0)
    omega = tf.Variable(1.5)
    with tf.GradientTape() as tape:
        samples = Nakagami(m, omega).sample(2048, seed=1)
        loss = tf.reduce_mean(samples)
    grads = tape.gradient(loss, [m, omega])
    for g in grads:
        assert g is not None
        assert tf.math.is_finite(g).numpy()
        assert tf.abs(g).numpy() > 0.0


def test_kl_self_is_zero():
    p = Nakagami(2.0, 1.3)
    kl = tfd.kl_divergence(p, p).numpy()
    np.testing.assert_allclose(kl, 0.0, atol=1e-6)


def test_kl_matches_gamma_kl():
    m1, o1 = 1.5, 1.2
    m2, o2 = 2.3, 0.8
    p = Nakagami(m1, o1)
    q = Nakagami(m2, o2)
    gp = tfd.Gamma(concentration=m1, rate=m1 / o1)
    gq = tfd.Gamma(concentration=m2, rate=m2 / o2)
    kl_nak = tfd.kl_divergence(p, q).numpy()
    kl_gam = tfd.kl_divergence(gp, gq).numpy()
    np.testing.assert_allclose(kl_nak, kl_gam, rtol=1e-6, atol=1e-7)


def test_kl_matches_monte_carlo():
    p = Nakagami(2.5, 1.2)
    q = Nakagami(1.7, 0.9)
    kl_analytic = tfd.kl_divergence(p, q).numpy()

    n = 200_000
    samples = p.sample(n, seed=42)
    mc = tf.reduce_mean(p.log_prob(samples) - q.log_prob(samples)).numpy()
    # MC standard error is small for n=200k; allow 3-sigma tolerance plus headroom.
    assert abs(kl_analytic - mc) < 0.01


@pytest.mark.parametrize("m,omega", [
    (0.5, 1.0),
    (1.0, 1.5),
    (2.5, 0.7),
    (4.0, 3.1),
])
def test_mean_matches_scipy(m, omega):
    d = Nakagami(m, omega)
    expected = scipy_nakagami.mean(m, scale=np.sqrt(omega))
    np.testing.assert_allclose(d.mean().numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("m,omega", [
    (0.5, 1.0),
    (1.0, 1.5),
    (2.5, 0.7),
    (4.0, 3.1),
])
def test_variance_and_stddev_match_scipy(m, omega):
    d = Nakagami(m, omega)
    expected_var = scipy_nakagami.var(m, scale=np.sqrt(omega))
    expected_std = scipy_nakagami.std(m, scale=np.sqrt(omega))
    np.testing.assert_allclose(d.variance().numpy(), expected_var, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(d.stddev().numpy(), expected_std, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("m,omega", [
    (0.5, 1.0),
    (1.0, 1.5),
    (2.5, 0.7),
    (4.0, 3.1),
])
def test_mode_matches_closed_form(m, omega):
    d = Nakagami(m, omega)
    expected = np.sqrt(max(0.0, (2.0 * m - 1.0) * omega / (2.0 * m)))
    np.testing.assert_allclose(d.mode().numpy(), expected, rtol=1e-5, atol=1e-6)


def test_square_returns_gamma_with_matched_parameters():
    m, omega = 2.3, 1.7
    d = Nakagami(m, omega)
    sq = d.square()
    assert isinstance(sq, tfd.Gamma)
    np.testing.assert_allclose(sq.concentration.numpy(), m)
    np.testing.assert_allclose(sq.rate.numpy(), m / omega)


def test_square_distribution_matches_monte_carlo():
    m, omega = 1.8, 2.1
    d = Nakagami(m, omega)
    sq = d.square()
    # Squared samples from Nakagami should look like Gamma draws.
    y = d.sample(200_000, seed=7)
    ysq = y * y
    np.testing.assert_allclose(
        tf.reduce_mean(ysq).numpy(), sq.mean().numpy(), rtol=1e-2)
    np.testing.assert_allclose(
        tf.math.reduce_variance(ysq).numpy(), sq.variance().numpy(), rtol=5e-2)


def test_kl_is_registered_analytically():
    p = Nakagami(1.5, 1.0)
    q = Nakagami(2.0, 1.5)
    # Should not hit the fallback path — calling kl_divergence directly
    # against the registered dispatch should return a concrete tensor.
    result = tfd.kl_divergence(p, q)
    assert tf.math.is_finite(result).numpy()
    assert result.numpy() >= 0.0


# The Wilson priors for structure factors should be Nakagami-equivalent to the
# previous HalfNormal / Weibull formulations after the refactor.

@pytest.mark.parametrize("epsilon,sigma", [(1.0, 1.0), (1.0, 2.5), (3.0, 0.7)])
def test_centric_wilson_factory_matches_half_normal(epsilon, sigma):
    from abismal.prior.structure_factor.wilson import centric_wilson
    dist = centric_wilson(epsilon, sigma)
    reference = tfd.HalfNormal(scale=tf.math.sqrt(epsilon * sigma))
    x = np.linspace(0.01, 5.0, 40).astype(np.float32)
    np.testing.assert_allclose(
        dist.log_prob(x).numpy(), reference.log_prob(x).numpy(),
        rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("epsilon,sigma", [(1.0, 1.0), (1.0, 2.5), (3.0, 0.7)])
def test_acentric_wilson_factory_matches_weibull(epsilon, sigma):
    from abismal.prior.structure_factor.wilson import acentric_wilson
    dist = acentric_wilson(epsilon, sigma)
    reference = tfd.Weibull(
        concentration=2.0, scale=tf.math.sqrt(epsilon * sigma))
    x = np.linspace(0.01, 5.0, 40).astype(np.float32)
    np.testing.assert_allclose(
        dist.log_prob(x).numpy(), reference.log_prob(x).numpy(),
        rtol=1e-4, atol=1e-5)


def test_wilson_distribution_dispatches_per_reflection():
    from abismal.prior.structure_factor.wilson import wilson_distribution
    centric = tf.constant([True, False, True, False])
    epsilon = tf.constant([1.0, 1.0, 2.0, 2.0])
    dist = wilson_distribution(centric, epsilon, 1.5)
    x = tf.constant([0.5, 0.5, 1.0, 1.0])
    lp = dist.log_prob(x).numpy()
    # The centric entries should match HalfNormal; acentric, Weibull(2, ...).
    expected = np.array([
        tfd.HalfNormal(scale=tf.math.sqrt(1.0 * 1.5)).log_prob(0.5).numpy(),
        tfd.Weibull(2.0, scale=tf.math.sqrt(1.0 * 1.5)).log_prob(0.5).numpy(),
        tfd.HalfNormal(scale=tf.math.sqrt(2.0 * 1.5)).log_prob(1.0).numpy(),
        tfd.Weibull(2.0, scale=tf.math.sqrt(2.0 * 1.5)).log_prob(1.0).numpy(),
    ])
    np.testing.assert_allclose(lp, expected, rtol=1e-4, atol=1e-5)
