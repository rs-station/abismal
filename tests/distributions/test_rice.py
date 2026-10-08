"""Tests for the Rice distribution's sampler.

Both normal components used to be drawn from the same stateless seed, which made
the second exactly the first minus nu. The samples were then a deterministic
function of a single normal draw and not Rice distributed, though their mean was
close enough that nothing downstream looked wrong.
"""
import numpy as np
import pytest
from scipy.stats import kstest
from scipy.stats import rice as scipy_rice

from abismal.distributions import Rice


@pytest.mark.parametrize("nu,sigma", [(0.0, 1.0), (2.0, 1.5), (5.0, 1.0)])
def test_samples_match_scipy(nu, sigma):
    z = Rice(nu, sigma).sample(20_000, seed=(1, 2)).numpy()
    # scipy parameterizes as rice(b=nu/sigma, scale=sigma)
    result = kstest(z, scipy_rice(nu / sigma, scale=sigma).cdf)
    assert result.pvalue > 1e-3


def test_components_are_independent():
    nu, sigma = 3.0, 1.0
    z = Rice(nu, sigma).sample(20_000, seed=(3, 4)).numpy()
    # E[z^2] = nu^2 + 2 sigma^2 for a Rice variable. With the shared seed it was
    # E[x^2 + (x - nu)^2] for x ~ N(nu, sigma), i.e. nu^2 + 2 sigma^2 as well,
    # so the second moment alone cannot tell them apart -- the variance can.
    expected_var = scipy_rice(nu / sigma, scale=sigma).var()
    assert np.isclose(z.var(), expected_var, rtol=0.05)
