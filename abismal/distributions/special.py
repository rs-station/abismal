"""Special functions that float32 torch does not provide stably."""
import math

import torch

# The asymptotic series below are accurate to float32 precision from here up.
# Smaller arguments are shifted up with the recurrence Gamma(x + 1) = x Gamma(x)
# rather than handed to torch.lgamma, whose float32 differences lose ~eps *
# lgamma(m) absolute -- ~1e-2 at m = 1e4 -- which the Nakagami variance,
# Omega (1 - Gamma(m + 1/2)^2 / (m Gamma(m)^2)) ~ Omega / (4 m), then amplifies.
_SHIFT = 4


def _series_excess(m):
    """Asymptotic series of lgamma(m + 1/2) - lgamma(m) - log(m) / 2, m >= 4."""
    r = 1.0 / m
    r2 = r * r
    return r * (-1.0 / 8.0 + r2 * (1.0 / 192.0 + r2 * (-1.0 / 640.0 + r2 * 17.0 / 14336.0)))


def log_gamma_half_ratio_excess(m):
    """``lgamma(m + 1/2) - lgamma(m) - log(m) / 2``, stable for all m > 0.

    This is the form to build moments of ``sqrt(Gamma(m, rate))`` from without
    cancellation: with ``r`` the value returned,

        E[sqrt(X)]   = exp(r) sqrt(m / rate)
        Var[sqrt(X)] = -expm1(2 r) m / rate.

    It tends to ``-1 / (8 m)`` as m grows.
    """
    large = m >= _SHIFT
    # Below the threshold, lgamma(m + 1/2) - lgamma(m) =
    #   [lgamma(m + 4 + 1/2) - lgamma(m + 4)] - sum_j log((m + j + 1/2) / (m + j)).
    m_small = torch.where(large, torch.ones_like(m), m)
    shifted = _series_excess(m_small + _SHIFT) + 0.5 * torch.log1p(_SHIFT / m_small)
    for j in range(_SHIFT):
        shifted = shifted - torch.log1p(0.5 / (m_small + j))
    m_large = torch.where(large, m, torch.full_like(m, float(_SHIFT)))
    return torch.where(large, _series_excess(m_large), shifted)


def log_gamma_half_ratio(m):
    """``lgamma(m + 1/2) - lgamma(m)``, stable for all m > 0 (TFP's
    ``-log_gamma_difference(0.5, m)``)."""
    return log_gamma_half_ratio_excess(m) + 0.5 * torch.log(m)


def log_gamma_stirling_correction(x):
    """``lgamma(x) - [(x - 1/2) log(x) - x + log(2 pi) / 2]``, stable for all x > 0.

    The remainder of Stirling's series, ~ 1 / (12 x). Densities written in terms of
    it avoid the cancellation between lgamma(x) and x log(x), both ~1e6 at
    x = 1e5, that costs a float32 log density ~0.1 absolute.
    """
    large = x >= _SHIFT
    x_large = torch.where(large, x, torch.full_like(x, float(_SHIFT)))
    x_small = torch.where(large, torch.ones_like(x), x)
    r = 1.0 / x_large
    r2 = r * r
    series = r * (
        1.0 / 12.0 + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0 + r2 * -1.0 / 1680.0))
    )
    direct = (
        torch.lgamma(x_small) - (x_small - 0.5) * torch.log(x_small) + x_small
        - 0.5 * math.log(2.0 * math.pi)
    )
    return torch.where(large, series, direct)
