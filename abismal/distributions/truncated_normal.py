import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

from abismal.distributions.implicit import implicit_reparameterization

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_HALF = math.log(0.5)


def _log1mexp(x):
    """``log(1 - exp(x))`` for ``x < 0`` (Maechler 2012)."""
    # Both branches see an argument for which they are finite, so neither
    # contributes NaN gradients through the where.
    near = x > -math.log(2.0)
    x_near = torch.where(near, x, torch.full_like(x, -1.0))
    x_far = torch.where(near, torch.full_like(x, -1.0), x)
    return torch.where(
        near, torch.log(-torch.expm1(x_near)), torch.log1p(-torch.exp(x_far))
    )


def _log_phi(x):
    return -0.5 * x * x - _HALF_LOG_2PI


def _log_ndtr(x):
    """``log Phi(x)``, with its derivative taken from :func:`_inverse_mills`.

    torch's own backward for ``log_ndtr`` is wrong far into the lower tail (it
    returns 0 at x = -1e6 and 0.399 at x = -1e13, where the slope is ~ -x).
    """
    finite = torch.isfinite(x)
    with torch.no_grad():
        value = torch.special.log_ndtr(x)
        # An infinite bound is a constant; an infinite slope would make the
        # spliced-in gradient term inf * 0 = NaN in the forward value.
        slope = torch.where(finite, _inverse_mills(x), torch.zeros_like(x))
    x_finite = torch.where(finite, x, torch.zeros_like(x))
    return implicit_reparameterization(value, (x_finite, slope))


def _inverse_mills(x):
    """``phi(x) / Phi(x)``, accurate in both tails."""
    # Below zero, sqrt(2/pi) / erfcx(-x/sqrt(2)) avoids dividing two underflowing
    # numbers. Above it, Phi(x) is no smaller than 1/2 and the ratio of exps is
    # fine, while erfcx(-x/sqrt(2)) overflows (x > ~13 in float32) and its
    # gradient turns to NaN.
    neg = x <= 0
    x_neg = torch.where(neg, x, torch.zeros_like(x))
    x_pos = torch.where(neg, torch.zeros_like(x), x)
    return torch.where(
        neg,
        math.sqrt(2.0 / math.pi) / torch.special.erfcx(-x_neg / math.sqrt(2.0)),
        torch.exp(_log_phi(x_pos) - torch.special.log_ndtr(x_pos)),
    )


def _inverse_log_ndtr(log_p, iterations=3):
    """Solve ``log_ndtr(v) = log_p`` for ``log_p <= log(1/2)``.

    The solve runs without a graph; one more Newton step with ``log_p`` live
    attaches the implicit gradient ``dv = dlog_p / (d log_ndtr / dv)``. Working in
    log space keeps the lower tail finite where ``ndtri(exp(log_p))`` would
    underflow to ``-inf`` (float32 ``Phi(v)`` underflows below ``v ~ -13``).
    """
    with torch.no_grad():
        lp = log_p.detach()
        tiny = torch.finfo(lp.dtype).tiny
        # ndtri is accurate wherever exp(log_p) is a normal float; past that, the
        # leading terms of log Phi(v) ~ -v^2/2 - log(-v) - log(2 pi)/2.
        deep = lp < math.log(tiny) + 5.0
        lp_deep = torch.where(deep, lp, torch.full_like(lp, -100.0))
        v_deep = -torch.sqrt(
            -2.0 * lp_deep - torch.log(-2.0 * lp_deep) - 2.0 * _HALF_LOG_2PI
        )
        v_ndtri = torch.special.ndtri(torch.exp(lp).clamp(tiny, 0.5))
        v = torch.where(deep, v_deep, v_ndtri)
        # log Phi is concave, so Newton converges monotonically from below.
        for _ in range(iterations):
            v = v - (torch.special.log_ndtr(v) - lp) / _inverse_mills(v)
        log_cdf = torch.special.log_ndtr(v)
        slope = _inverse_mills(v)
    return v - (log_cdf - log_p) / slope


class TruncatedNormal(Distribution):
    """The normal distribution ``N(loc, scale)`` truncated to ``[low, high]``.

    Everything is computed on whichever side of the interval's midpoint lies in
    the lower tail of the standardized normal (reflecting the interval when it
    lies above the mean), where ``Phi`` and ``log Phi`` keep their relative
    precision. ``rsample`` inverts the CDF and is differentiable by autograd.
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    has_rsample = True

    def __init__(self, loc, scale, low, high, validate_args=False):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _standardize_bound(self, bound):
        # An infinite bound stays infinite and constant: (inf - loc) / scale has
        # an infinite derivative in scale, and inf * 0 is NaN in the backward pass.
        finite = torch.isfinite(bound)
        bound_safe = torch.where(finite, bound, torch.zeros_like(bound))
        return torch.where(finite, (bound_safe - self.loc) / self.scale, bound)

    def _standardized(self):
        """Return ``(flip, lo, hi, log_z)`` of the reflected, standardized interval.

        ``flip`` marks intervals reflected through the mean, ``[lo, hi]`` has
        ``lo + hi <= 0`` and ``log_z = log(Phi(hi) - Phi(lo))``.
        """
        a = self._standardize_bound(self.low)
        b = self._standardize_bound(self.high)
        flip = (a + b) > 0
        lo = torch.where(flip, -b, a)
        hi = torch.where(flip, -a, b)
        log_cdf_hi = _log_ndtr(hi)
        log_z = log_cdf_hi + _log1mexp(_log_ndtr(lo) - log_cdf_hi)
        return flip, lo, hi, log_z

    def _unstandardize(self, flip, v):
        return self.loc + self.scale * torch.where(flip, -v, v)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        *_, log_z = self._standardized()
        x = (value - self.loc) / self.scale
        log_p = _log_phi(x) - torch.log(self.scale) - log_z
        inside = (value >= self.low) & (value <= self.high)
        return torch.where(inside, log_p, torch.full_like(log_p, -math.inf))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        flip, lo, hi, log_z = self._standardized()
        with torch.no_grad():
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            u = u.clamp(min=torch.finfo(u.dtype).tiny)
        log_cdf_lo = _log_ndtr(lo)
        log_cdf_hi = _log_ndtr(hi)
        # The target CDF value p = Phi(lo) + u Z, as log p ...
        r = torch.exp(log_cdf_lo - log_cdf_hi)
        log_p = log_cdf_hi + torch.log(r + u * (1.0 - r))
        # ... or, past the median, its complement 1 - p = Phi(-hi) + (1 - u) Z,
        # which does not round to zero as p rounds to one.
        log_q = torch.logaddexp(
            _log_ndtr(-hi), torch.log1p(-u) + log_z
        )
        lower = log_p <= _LOG_HALF
        v = _inverse_log_ndtr(torch.where(lower, log_p, log_q))
        v = torch.where(lower, v, -v)
        # Round-off can land a hair outside the interval.
        return torch.clamp(self._unstandardize(flip, v), self.low, self.high)

    def _mills_terms(self):
        """``(flip, lo, hi, phi(lo) / Z, phi(hi) / Z)``, infinite bounds zeroed."""
        flip, lo, hi, _ = self._standardized()
        log_cdf_lo = _log_ndtr(lo)
        log_cdf_hi = _log_ndtr(hi)
        # With r = Phi(lo) / Phi(hi) and the inverse Mills ratio
        # lambda(x) = phi(x) / Phi(x), phi(hi) / Z = lambda(hi) / (1 - r) and
        # phi(lo) / Z = lambda(lo) r / (1 - r). Deep in the tail this keeps the
        # precision that exp(log phi - log Z) loses (~eps * |log Z| relative).
        log_r = log_cdf_lo - log_cdf_hi
        one_minus_r = -torch.expm1(log_r)
        # Infinite bounds contribute nothing; keep inf * 0 out of the products.
        lo_finite, hi_finite = torch.isfinite(lo), torch.isfinite(hi)
        zero = torch.zeros_like(lo)
        lo_safe = torch.where(lo_finite, lo, zero)
        hi_safe = torch.where(hi_finite, hi, zero)
        lam_lo = _inverse_mills(lo_safe)
        lam_hi = _inverse_mills(hi_safe)
        rho_lo = torch.where(lo_finite, lam_lo * torch.exp(log_r) / one_minus_r, zero)
        rho_hi = torch.where(hi_finite, lam_hi / one_minus_r, zero)
        return flip, lo_safe, hi_safe, rho_lo, rho_hi

    @property
    def mean(self):
        flip, _, _, rho_lo, rho_hi = self._mills_terms()
        return self._unstandardize(flip, rho_lo - rho_hi)

    @property
    def variance(self):
        _, lo, hi, rho_lo, rho_hi = self._mills_terms()
        m = rho_lo - rho_hi
        # This cancels when the whole interval is many sigma into a tail (float32
        # loses ~all digits by |hi| ~ 100); clamp so stddev stays real there.
        var = 1.0 + lo * rho_lo - hi * rho_hi - m * m
        return self.scale * self.scale * torch.clamp(var, min=0.0)
