import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

from abismal.distributions.implicit import implicit_reparameterization


def folded_normal_sample_gradients(z, loc, scale):
    """Implicit reparameterization gradients ``(dz/dloc, dz/dscale)`` of a folded normal.

    With ``p_a = N(z|loc,scale)`` and ``p_b = N(z|-loc,scale)``,

        dz/dloc   = (p_a - p_b) / (p_a + p_b)
        dz/dscale = [(z - loc) p_a + (z + loc) p_b] / [scale (p_a + p_b)].

    The density ratio is ``p_a / p_b = exp(2 z loc / scale^2)``, so the first is
    ``tanh(z loc / scale^2)`` and the second ``(z - loc dz/dloc) / scale``. Written
    this way neither ever divides two underflowed densities, which the direct
    form does far out in the tail (0/0 = NaN).
    """
    dloc = torch.tanh(z * loc / (scale * scale))
    dscale = (z - loc * dloc) / scale
    return dloc, dscale


class FoldedNormal(Distribution):
    """The folded normal distribution, the law of ``|x|`` for ``x ~ N(loc, scale)``.

    ``rsample`` uses implicit reparameterization gradients.
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    # Above this |loc/scale| the folded mass is negligible (< 1e-23) and the
    # moments are those of the normal; the folded expressions for them cancel
    # catastrophically there.
    _normal_crossover = 10.0

    def __init__(self, loc, scale, validate_args=False):
        self.loc, self.scale = broadcast_all(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FoldedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(FoldedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        loc, scale = self.loc, self.scale
        with torch.no_grad():
            eps = torch.randn(shape, dtype=loc.dtype, device=loc.device)
            z = torch.abs(loc + scale * eps)
            dloc, dscale = folded_normal_sample_gradients(z, loc, scale)
        return implicit_reparameterization(z, (loc, dloc), (scale, dscale))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        loc, scale = self.loc, self.scale
        log_norm = -torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
        a = (value - loc) / scale
        b = (value + loc) / scale
        return torch.logaddexp(-0.5 * a * a, -0.5 * b * b) + log_norm

    def cdf(self, value):
        # F(x) = Phi((x - loc)/scale) + Phi((x + loc)/scale) - 1, the *sum* of two
        # erfs. The TF version read `erf(a) - erf(b)` until 2026-08-24.
        loc, scale = self.loc, self.scale
        value = torch.as_tensor(value, dtype=loc.dtype, device=loc.device)
        ir2 = 1.0 / math.sqrt(2.0)
        cdf = 0.5 * (
            torch.erf(ir2 * (value + loc) / scale)
            + torch.erf(ir2 * (value - loc) / scale)
        )
        # The support is non-negative; the expression runs negative below zero.
        return torch.where(value < 0, torch.zeros_like(cdf), cdf)

    @staticmethod
    def _folded_normal_mean(loc, scale):
        c = loc / scale
        return (
            scale * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * c * c)
            + loc * torch.erf(c / math.sqrt(2.0))
        )

    def _crossover_and_safe_mean(self):
        loc, scale = self.loc, self.scale
        idx = torch.abs(loc / scale) >= self._normal_crossover
        # Substitute harmless values in the branch torch.where discards, so its
        # gradient is zero rather than NaN.
        one = torch.ones_like(loc)
        loc_safe = torch.where(idx, one, loc)
        scale_safe = torch.where(idx, one, scale)
        return idx, self._folded_normal_mean(loc_safe, scale_safe)

    @property
    def mean(self):
        idx, m = self._crossover_and_safe_mean()
        return torch.where(idx, torch.abs(self.loc), m)

    @property
    def variance(self):
        loc, scale = self.loc, self.scale
        idx, m = self._crossover_and_safe_mean()
        return torch.where(idx, scale * scale, loc * loc + scale * scale - m * m)
