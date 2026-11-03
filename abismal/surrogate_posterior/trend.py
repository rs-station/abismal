import numpy as np
import torch
from abismal import tracking


class ResolutionTrend(torch.nn.Module):
    """
    A learned falloff T(d) that multiplies the surrogate posterior's intensities on their way to the likelihood and
    the output, while the posterior and its KL stay on a flat scale:

        I_obs ~ s * T(d) * F**2,    F ~ q,    KL[q(F) || flat prior]

    This is the same model as a prior whose Σ follows T(d), with the falloff in a few parameters instead of spread over
    every reflection's posterior (a non-centered parameterization), so that it can trade quickly against the scales.

    kind='wilson': log T = log k - b / d**2.  kind='spline': log T piecewise linear in 1/d**2, with `knots` knots at
    quantiles of the reflections' 1/d**2. Parameters are in units of 0.1 of log T (of b for 'wilson'), because Adam
    moves a parameter about the learning rate per step.
    """
    def __init__(self, rac, kind='wilson', knots=16):
        super().__init__()
        if kind not in ('wilson', 'spline'):
            raise ValueError(f"kind must be 'wilson' or 'spline', not {kind!r}")
        self.rac = rac
        self.kind = kind
        q = torch.as_tensor(rac.dHKL, dtype=torch.float32).reshape(-1) ** -2
        if kind == 'wilson':
            self.register_buffer('q', q)
            self._log_k = torch.nn.Parameter(torch.tensor(0.))
            self._b = torch.nn.Parameter(torch.tensor(0.))
        else:
            edges = np.unique(np.quantile(q.numpy(), np.linspace(0, 1, knots)))
            knot = torch.as_tensor(edges, dtype=torch.float32)
            index = torch.bucketize(q, knot, right=True).clamp(1, len(knot) - 1) - 1
            frac = (q - knot[index]) / (knot[index + 1] - knot[index])
            self.register_buffer('knots', knot)
            self.register_buffer('index', index)
            self.register_buffer('frac', frac)
            self._values = torch.nn.Parameter(torch.zeros(len(knot)))

    def log_trend(self):
        """log T of every reflection in the ASU, by Miller id."""
        if self.kind == 'wilson':
            return 10. * (self._log_k - self._b * self.q)
        v = 10. * self._values
        return (1. - self.frac) * v[self.index] + self.frac * v[self.index + 1]

    def forward(self, asu_id, hkl):
        """T(d) of the given reflections, shape (reflections, 1)."""
        if self.kind == 'wilson':
            tracking.add_metric(10. * self._b.detach() * 2., 'TrendB')
            tracking.add_metric(torch.exp(10. * self._log_k.detach()), 'TrendK')
        else:
            v = 10. * self._values.detach()
            tracking.add_metric(v[0] - v[-1], 'TrendRange')
        return torch.exp(self.rac.gather(self.log_trend(), asu_id, hkl)).reshape(-1, 1)
