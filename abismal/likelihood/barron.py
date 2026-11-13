"""
Barron's general robust distribution as the likelihood's shape (see abismal.robust):

    log p(I_obs | I_pred) = -ρ(z, α) - log σ - log Z(α),   z = (I_pred - I_obs) / σ

α = 2 is the normal, α = 1 the pseudo-Huber (normal at the center, Laplace-like tails, bounded influence) and α = 0
the Cauchy. α is fixed, or learned on the ELBO within [alpha_min, 2) through a sigmoid.
"""
import math
import torch
from abismal import robust, tracking
from abismal.likelihood.location_scale import LocationScale


class BarronShape(torch.nn.Module):
    # α stays this far below 2 when learned: there ρ's general form loses float32 precision, and the normal is the
    # limit anyway.
    ALPHA_MARGIN = 1e-3

    def __init__(self, alpha=1., learn_alpha=False, alpha_min=0.):
        super().__init__()
        # log Z(α)'s spline, built here (eagerly) and carried as buffers so it follows the module's device.
        grid, c = robust.spline_coefficients()
        self.register_buffer('spline_grid', torch.as_tensor(grid, dtype=torch.float32), persistent=False)
        self.register_buffer('spline_c', torch.as_tensor(c, dtype=torch.float32), persistent=False)
        if not 0. <= alpha <= 2.:
            raise ValueError(f"alpha must be in [0, 2], not {alpha}")
        self.learn_alpha = learn_alpha
        if learn_alpha:
            self.alpha_min = float(alpha_min)
            self.alpha_max = 2. - self.ALPHA_MARGIN
            if not self.alpha_min < alpha < self.alpha_max:
                raise ValueError(f"a learned alpha must start inside ({alpha_min}, {self.alpha_max}), not {alpha}")
            u = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
            self.alpha_logit = torch.nn.Parameter(torch.tensor(math.log(u / (1. - u))))
        else:
            self._alpha = float(alpha)

    @property
    def alpha(self):
        if self.learn_alpha:
            return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_logit)
        return self._alpha

    def log_prob(self, x, loc, scale):
        z = (x - loc) / scale
        coefficients = (self.spline_grid.to(z.dtype), self.spline_c.to(z.dtype))
        return robust.logpdf(z, self.alpha, coefficients) - torch.log(scale)

    @torch.no_grad()
    def register_metrics(self):
        tracking.add_metric(torch.as_tensor(self.alpha, dtype=torch.float32, device=self.spline_grid.device), 'alpha')


class BarronLikelihood(LocationScale):
    """Barron's distribution with the integration program's σ as its scale."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, ipred, iobs, sigiobs):
        return self.shape.log_prob(ipred, iobs, sigiobs)

    @torch.no_grad()
    def register_metrics(self, ipred, iobs, sigiobs):
        super().register_metrics(ipred, iobs, sigiobs)
        self.shape.register_metrics()
