"""
Barron's general robust distribution (J. T. Barron, "A General and Adaptive Robust Loss Function", CVPR 2019,
arXiv:1701.03077) for abismal's likelihoods (the VI model's and abismal.ml's), restricted to the shapes 0 <= α <= 2 where it is a normalizable
density:

    ρ(z, α) = |α - 2| / α · ((z² / |α - 2| + 1)^(α/2) - 1)          log p(x) = -ρ(x/c, α) - log c - log Z(α)

α = 2 is the normal distribution (ρ = z²/2, c its standard deviation), α = 1 the pseudo-Huber / Charbonnier loss
(bounded influence) and α = 0 the Cauchy (ρ = log(1 + z²/2)). For 0 < α < 2 the tails fall off as exp(-|z|^α):
lighter than any Student t, with every moment finite. The influence ∂ρ/∂z = z·w(z) with the IRLS weight
w(z) = (z²/|α - 2| + 1)^(α/2 - 1), which grows without bound for α > 1, tends to a constant at α = 1 and redescends
to 0 for α < 1.

Z(α) = ∫ exp(-ρ(x, α)) dx has no closed form; it is integrated numerically once on a grid of α in [0, 2] and
interpolated with a cubic spline in log Z. Written for abismal; Barron's own PyTorch port (jonbarron/
robust_loss_pytorch) was used as a reference for the special cases at α = 0 and 2.
"""
import math
import numpy as np
import torch

_EPS = float(np.finfo(np.float32).eps)
_GRID = 401  # α from 0 to 2 in steps of 0.005


def rho(z, alpha):
    """Barron's ρ(z, α) for 0 <= α <= 2, elementwise; `alpha` a number or a tensor broadcasting with `z`."""
    alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    z2 = z * z
    beta = torch.clamp(torch.abs(alpha - 2.), min=_EPS)
    a = torch.clamp(alpha, min=_EPS)
    general = beta / a * (torch.pow(z2 / beta + 1., 0.5 * alpha) - 1.)
    return torch.where(alpha == 0., torch.log1p(0.5 * z2), torch.where(alpha == 2., 0.5 * z2, general))


def weight(z, alpha):
    """The IRLS weight (∂ρ/∂z)/z = (z²/|α - 2| + 1)^(α/2 - 1)."""
    alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    beta = torch.clamp(torch.abs(alpha - 2.), min=_EPS)
    return torch.where(alpha == 2., torch.ones_like(z), torch.pow(z * z / beta + 1., 0.5 * alpha - 1.))


def _rho_np(x, alpha):
    if alpha == 0.:
        return np.log1p(0.5 * x * x)
    if alpha == 2.:
        return 0.5 * x * x
    beta = abs(alpha - 2.)
    return beta / alpha * ((x * x / beta + 1.) ** (0.5 * alpha) - 1.)


def partition(alpha):
    """Z(α) by numerical integration (double precision); exact at α = 0 (π√2) and α = 2 (√(2π))."""
    from scipy.integrate import quad
    if alpha == 0.:
        return math.pi * math.sqrt(2.)
    if alpha == 2.:
        return math.sqrt(2. * math.pi)
    # Symmetric: twice the integral over [0, ∞), split at 1 for the peak.
    f = lambda x: math.exp(-_rho_np(x, alpha))
    return 2. * (quad(f, 0., 1., epsabs=0., epsrel=1e-12, limit=200)[0]
                 + quad(f, 1., np.inf, epsabs=0., epsrel=1e-12, limit=500)[0])


_spline = {}


def spline_coefficients():
    """The grid of α and the cubic spline's coefficients for log Z(α), as numpy arrays (computed once)."""
    if 'np' not in _spline:
        from scipy.interpolate import CubicSpline
        grid = np.linspace(0., 2., _GRID)
        cs = CubicSpline(grid, np.log([partition(float(a)) for a in grid]))
        _spline['np'] = (grid, cs.c)  # c[k, i]: coefficient of (α - grid[i])^(3 - k)
    return _spline['np']


@torch.compiler.disable
def _coefficients(device, dtype):
    key = (str(device), dtype)
    if key not in _spline:
        grid, c = spline_coefficients()
        _spline[key] = (torch.as_tensor(grid, device=device, dtype=dtype), torch.as_tensor(c, device=device, dtype=dtype))
    return _spline[key]


def log_partition(alpha, like=None, coefficients=None):
    """
    log Z(α) for 0 <= α <= 2, differentiable in α (cubic spline through a numerically integrated grid).
    `coefficients`, (grid, c) tensors from spline_coefficients, skips the lookup (for compiled code, which should
    not build the spline).
    """
    ref = like if like is not None else (alpha if torch.is_tensor(alpha) else torch.zeros(()))
    alpha = torch.as_tensor(alpha, dtype=ref.dtype, device=ref.device)
    grid, c = coefficients if coefficients is not None else _coefficients(ref.device, ref.dtype)
    h = grid[1] - grid[0]
    i = torch.clamp(torch.floor(alpha / h).long(), 0, _GRID - 2)
    t = alpha - grid[i]
    return ((c[0, i] * t + c[1, i]) * t + c[2, i]) * t + c[3, i]


def logpdf(z, alpha, coefficients=None):
    """log density of the standardized Barron distribution at z (scale 1)."""
    return -rho(z, alpha) - log_partition(alpha, like=z, coefficients=coefficients)
