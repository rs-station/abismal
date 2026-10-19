"""The Wilson priors on structure factors are Nakagami distributions, which must
agree with the HalfNormal (centric) and Weibull (acentric) forms they replaced."""
import numpy as np
import pytest
import torch
from scipy.stats import halfnorm, weibull_min

from abismal.prior.structure_factor.wilson import (
    acentric_wilson,
    centric_wilson,
    wilson_distribution,
)

X = np.linspace(0.01, 5.0, 40).astype(np.float32)


@pytest.mark.parametrize("epsilon,sigma", [(1.0, 1.0), (1.0, 2.5), (3.0, 0.7)])
def test_centric_wilson_factory_matches_half_normal(epsilon, sigma):
    dist = centric_wilson(torch.tensor(epsilon), sigma)
    expected = halfnorm.logpdf(X, scale=np.sqrt(epsilon * sigma))
    np.testing.assert_allclose(
        dist.log_prob(torch.as_tensor(X)).numpy(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("epsilon,sigma", [(1.0, 1.0), (1.0, 2.5), (3.0, 0.7)])
def test_acentric_wilson_factory_matches_weibull(epsilon, sigma):
    dist = acentric_wilson(torch.tensor(epsilon), sigma)
    expected = weibull_min.logpdf(X, 2.0, scale=np.sqrt(epsilon * sigma))
    np.testing.assert_allclose(
        dist.log_prob(torch.as_tensor(X)).numpy(), expected, rtol=1e-4, atol=1e-5)


def test_wilson_distribution_dispatches_per_reflection():
    centric = torch.tensor([True, False, True, False])
    epsilon = torch.tensor([1.0, 1.0, 2.0, 2.0])
    dist = wilson_distribution(centric, epsilon, 1.5)
    x = torch.tensor([0.5, 0.5, 1.0, 1.0])
    lp = dist.log_prob(x).numpy()
    # The centric entries should match HalfNormal; acentric, Weibull(2, ...).
    expected = np.array([
        halfnorm.logpdf(0.5, scale=np.sqrt(1.0 * 1.5)),
        weibull_min.logpdf(0.5, 2.0, scale=np.sqrt(1.0 * 1.5)),
        halfnorm.logpdf(1.0, scale=np.sqrt(2.0 * 1.5)),
        weibull_min.logpdf(1.0, 2.0, scale=np.sqrt(2.0 * 1.5)),
    ])
    np.testing.assert_allclose(lp, expected, rtol=1e-4, atol=1e-5)
