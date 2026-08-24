"""Tests for the folded normal distribution.

The CDF is checked against scipy because it went wrong quietly: it read
`erf(a) - erf(b)` where the folded normal CDF is the sum, and nothing in abismal
calls cdf(), so no result ever disagreed with anything. The sampler's
implicit-reparameterization gradients are derived from the *correct* CDF, so the
two were inconsistent with each other -- which is what these tests pin down.
"""
import numpy as np
import pytest
import tensorflow as tf
from scipy.stats import foldnorm as scipy_foldnorm

from abismal.distributions import FoldedNormal


# scipy parameterizes as foldnorm(c=loc/scale, scale=scale)
PARAMS = [
    (0.0, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (3.0, 0.5),
    (0.2, 2.5),
]


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_matches_scipy(loc, scale):
    x = np.linspace(0.0, loc + 6.0 * scale, 50).astype("float32")
    result = FoldedNormal(
        tf.constant(loc, tf.float32), tf.constant(scale, tf.float32)
    ).cdf(x).numpy()
    expected = scipy_foldnorm.cdf(x / scale, loc / scale)
    assert np.allclose(result, expected, atol=1e-5), (
        f"max deviation {np.abs(result - expected).max():.4g}"
    )


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_is_a_distribution_function(loc, scale):
    """Non-decreasing, in [0, 1], and zero below the support."""
    x = np.linspace(-3.0 * scale, loc + 8.0 * scale, 200).astype("float32")
    cdf = FoldedNormal(
        tf.constant(loc, tf.float32), tf.constant(scale, tf.float32)
    ).cdf(x).numpy()

    assert np.all(cdf >= 0.0), f"negative CDF, min {cdf.min():.4g}"
    assert np.all(cdf <= 1.0 + 1e-6), f"CDF above 1, max {cdf.max():.4g}"
    assert np.all(np.diff(cdf) >= -1e-6), "CDF is not non-decreasing"
    assert np.all(cdf[x < 0] == 0.0), "CDF is non-zero below the support"


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_agrees_with_the_sampler(loc, scale):
    """The empirical CDF of the samples must follow cdf().

    cdf() and _sample_n are independent code paths, and it was their
    disagreement that the sign error amounted to.
    """
    dist = FoldedNormal(tf.constant(loc, tf.float32), tf.constant(scale, tf.float32))
    samples = dist.sample(200000, seed=0).numpy()
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        x = np.quantile(samples, q)
        assert dist.cdf(np.float32(x)).numpy() == pytest.approx(q, abs=0.01)


@pytest.mark.parametrize("loc,scale", PARAMS)
def test_cdf_is_the_integral_of_the_density(loc, scale):
    """cdf() against a numerical integral of exp(log_prob), the other direction."""
    dist = FoldedNormal(tf.constant(loc, tf.float32), tf.constant(scale, tf.float32))
    x = np.linspace(0.0, loc + 8.0 * scale, 20001).astype("float32")
    pdf = np.exp(dist.log_prob(x).numpy())
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))])
    assert np.allclose(dist.cdf(x).numpy(), integral, atol=2e-4)
