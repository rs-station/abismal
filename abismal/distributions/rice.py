import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

from abismal.distributions.implicit import implicit_reparameterization


# Above this argument 1 - I1(x)/I0(x) comes from its asymptotic series (relative
# error < 1e-7 from here up). torch's i1e(x)/i0e(x) is off by an ulp or two there
# -- it exceeds 1 for x >~ 1e9 -- and dz/dsigma subtracts nu times it from z.
_BESSEL_RATIO_SERIES = 30.0


def _one_minus_bessel_ratio(x):
    """``1 - I1(x)/I0(x)`` for ``x >= 0``."""
    large = x >= _BESSEL_RATIO_SERIES
    x_large = torch.where(large, x, torch.full_like(x, _BESSEL_RATIO_SERIES))
    x_small = torch.where(large, torch.zeros_like(x), x)
    r = 1.0 / x_large
    series = r * (0.5 + r * (1.0 / 8.0 + r * (1.0 / 8.0 + r * (25.0 / 128.0 + r * 13.0 / 32.0))))
    direct = 1.0 - torch.special.i1e(x_small) / torch.special.i0e(x_small)
    return torch.where(large, series, direct)


def rice_sample_gradients(z, nu, sigma):
    """Implicit reparameterization gradients ``(dz/dnu, dz/dsigma)`` of a Rice variable.

    dz/dnu = I1(x) / I0(x) with x = nu z / sigma^2. Rice is a scale family in
    (nu, sigma), z = sigma w(nu/sigma), which gives
    dz/dsigma = (z - nu dz/dnu) / sigma. That difference is formed as
    (z - |nu|) + |nu| (1 - I1/I0), both terms accurate, because at high SNR
    z - nu I1/I0 is a small difference of two large numbers.
    """
    x = torch.abs(nu) * z / (sigma * sigma)
    s = _one_minus_bessel_ratio(x)
    dnu = torch.sign(nu) * (1.0 - s)
    dsigma = ((z - torch.abs(nu)) + torch.abs(nu) * s) / sigma
    return dnu, dsigma


def _laguerre_half(x):
    """The Laguerre polynomial L_{1/2}(x) for x <= 0.

    L_{1/2}(x) = exp(x/2) [(1 - x) I0(-x/2) - x I1(-x/2)], and for x <= 0 the
    exp(x/2) is exactly the scaling of i0e/i1e at -x/2.
    """
    y = -0.5 * x
    return (1.0 - x) * torch.special.i0e(y) - x * torch.special.i1e(y)


class Rice(Distribution):
    """The Rice distribution, the law of ``|x|`` for ``x ~ N([nu, 0], sigma^2 I)``.

    ``rsample`` uses implicit reparameterization gradients.
    """
    arg_constraints = {"nu": constraints.real, "sigma": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    # Above this |nu/sigma| the moments are swapped for those of N(|nu|, sigma);
    # the variance expression cancels catastrophically there.
    _normal_crossover = 40.0

    def __init__(self, nu, sigma, validate_args=False):
        self.nu, self.sigma = broadcast_all(nu, sigma)
        super().__init__(self.nu.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Rice, _instance)
        batch_shape = torch.Size(batch_shape)
        new.nu = self.nu.expand(batch_shape)
        new.sigma = self.sigma.expand(batch_shape)
        super(Rice, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        nu, sigma = self.nu, self.sigma
        with torch.no_grad():
            # Two independent draws. Sharing one made the second component the
            # first minus nu, and the result was not Rice distributed.
            eps = torch.randn((2,) + tuple(shape), dtype=nu.dtype, device=nu.device)
            z1 = nu + sigma * eps[0]
            z2 = sigma * eps[1]
            z = torch.sqrt(z1 * z1 + z2 * z2)
            dnu, dsigma = rice_sample_gradients(z, nu, sigma)
        return implicit_reparameterization(z, (nu, dnu), (sigma, dsigma))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        nu, sigma = self.nu, self.sigma
        positive = value > 0
        x = torch.where(positive, value, torch.ones_like(value))
        s2 = sigma * sigma
        # log I0(t) = log i0e(t) + |t|, folded into the quadratic:
        # -(x^2 + nu^2)/2s^2 + |x nu|/s^2 = -(x - |nu|)^2 / 2s^2 for x > 0. The
        # unfolded form loses everything to cancellation at large SNR in float32.
        d = x - torch.abs(nu)
        log_p = (
            torch.log(x) - torch.log(s2) - 0.5 * d * d / s2
            + torch.log(torch.special.i0e(x * nu / s2))
        )
        return torch.where(positive, log_p, torch.full_like(log_p, -math.inf))

    def _crossover_and_safe_laguerre(self):
        nu, sigma = self.nu, self.sigma
        snr = nu / sigma
        idx = torch.abs(snr) > self._normal_crossover
        snr_safe = torch.where(idx, torch.zeros_like(snr), snr)
        return idx, _laguerre_half(-0.5 * snr_safe * snr_safe)

    @property
    def mean(self):
        nu, sigma = self.nu, self.sigma
        idx, lag = self._crossover_and_safe_laguerre()
        mean = sigma * math.sqrt(0.5 * math.pi) * lag
        return torch.where(idx, torch.abs(nu), mean)

    @property
    def variance(self):
        nu, sigma = self.nu, self.sigma
        idx, lag = self._crossover_and_safe_laguerre()
        s2 = sigma * sigma
        variance = 2.0 * s2 + nu * nu - 0.5 * math.pi * s2 * lag * lag
        return torch.where(idx, s2, variance)
