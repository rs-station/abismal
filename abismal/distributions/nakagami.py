import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.kl import kl_divergence, register_kl
from torch.distributions.utils import broadcast_all

from abismal.distributions.special import (
    log_gamma_half_ratio_excess,
    log_gamma_stirling_correction,
)


class Nakagami(Distribution):
    """The Nakagami distribution.

    If ``X ~ Gamma(concentration=m, rate=m/Omega)`` then
    ``Y = sqrt(X) ~ Nakagami(m, Omega)``. ``rsample`` takes the square root of a
    reparameterized Gamma sample (torch's implicit Gamma gradients).

    The density is

        f(y; m, Omega) = 2 m^m / (Gamma(m) Omega^m) y^{2m - 1}
                         exp(-m y^2 / Omega),    y >= 0, m >= 1/2, Omega > 0.

    Special cases relevant to Wilson's distributions of X-ray structure
    factor amplitudes:

      * ``Nakagami(m=0.5, Omega=sigma**2)`` is ``HalfNormal(scale=sigma)``
        (the centric Wilson prior).
      * ``Nakagami(m=1.0, Omega=sigma**2)`` is
        ``Weibull(concentration=2., scale=sigma)`` (the acentric Wilson prior).
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "spread": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, concentration, spread, validate_args=False):
        self.concentration, self.spread = broadcast_all(concentration, spread)
        super().__init__(self.concentration.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Nakagami, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.spread = self.spread.expand(batch_shape)
        super(Nakagami, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def square(self):
        """Return the Gamma distribution of the squared random variable.

        If ``Y ~ Nakagami(m, Omega)`` then ``Y**2 ~ Gamma(m, m / Omega)``.
        """
        m = self.concentration
        return Gamma(m, m / self.spread, validate_args=self._validate_args)

    def rsample(self, sample_shape=torch.Size()):
        return torch.sqrt(self.square().rsample(sample_shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        m, omega = self.concentration, self.spread
        positive = value > 0
        y = torch.where(positive, value, torch.ones_like(value))
        # log f = log 2 + m log(m / Omega) - lgamma(m) + (2m - 1) log y - m y^2 / Omega,
        # regrouped with Stirling's series so that no two O(m log m) terms cancel:
        #   log 2 - log y + m (log1p(u) - u) + log(m) / 2 - log(2 pi) / 2 - c(m),
        # u = y^2 / Omega - 1, c the Stirling remainder.
        u = y * y / omega - 1.0
        # log t, t = y^2 / Omega: log1p(u) near the mode, where it is the
        # accurate one, and directly away from it, where u rounds towards -1.
        near = torch.abs(u) < 0.5
        log_t = torch.where(
            near,
            torch.log1p(torch.where(near, u, torch.zeros_like(u))),
            2.0 * torch.log(y) - torch.log(omega),
        )
        log_p = (
            math.log(2.0) - torch.log(y) + m * (log_t - u)
            + 0.5 * torch.log(m) - 0.5 * math.log(2.0 * math.pi)
            - log_gamma_stirling_correction(m)
        )
        return torch.where(positive, log_p, torch.full_like(log_p, -math.inf))

    def cdf(self, value):
        m, omega = self.concentration, self.spread
        value = torch.as_tensor(value, dtype=m.dtype, device=m.device)
        y = torch.clamp(value, min=0.0)
        return torch.special.gammainc(m, m * y * y / omega)

    @property
    def mean(self):
        # sqrt(Omega / m) Gamma(m + 1/2) / Gamma(m)
        return torch.sqrt(self.spread) * torch.exp(
            log_gamma_half_ratio_excess(self.concentration)
        )

    @property
    def variance(self):
        # Omega (1 - Gamma(m + 1/2)^2 / (m Gamma(m)^2)) = -Omega expm1(2 r)
        return -self.spread * torch.expm1(
            2.0 * log_gamma_half_ratio_excess(self.concentration)
        )

    @property
    def mode(self):
        m, omega = self.concentration, self.spread
        return torch.sqrt(torch.clamp((m - 0.5) * omega / m, min=0.0))


@register_kl(Nakagami, Nakagami)
def _kl_nakagami_nakagami(p, q):
    """KL divergence is invariant under the bijection y -> y^2, so
    KL(Nakagami_p || Nakagami_q) = KL(Gamma_p || Gamma_q), which torch has."""
    return kl_divergence(p.square(), q.square())
