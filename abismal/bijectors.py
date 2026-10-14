"""
Maps from the real line to the positive reals, for parameters that must be positive.
"""
import torch


def squareplus(x):
    """
    squareplus(x) = (x + sqrt(x^2 + 4)) / 2  (Barron, 2021).

    Smooth and increasing, with squareplus(0) = 1 and slope 1/2 there. It is
    linear for large x, like softplus, but approaches zero algebraically, about
    1/|x|, where softplus and exp approach it exponentially. Small values are
    therefore reached only slowly and keep a usable gradient.

    For x < 0 the direct form cancels catastrophically in floating point (in
    float32 it would round to zero near x = -1e4), so the equal expression
    2 / (sqrt(x^2 + 4) - x) is used there.
    """
    r = torch.sqrt(x * x + 4.)
    return torch.where(x >= 0, 0.5 * (x + r), 2. / (r - x))


def squareplus_inverse(y):
    """The inverse of `squareplus`, for y > 0: y - 1/y."""
    return y - 1. / y


BIJECTORS = {
    'exp': (torch.exp, torch.log),
    'squareplus': (squareplus, squareplus_inverse),
}
