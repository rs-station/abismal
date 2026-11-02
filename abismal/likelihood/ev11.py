import math
import torch
from torch import distributions as td
from abismal import tracking
from abismal.likelihood.location_scale import LocationScale


class Ev11Likelihood(LocationScale):
    """
    A normal (or Student-t) likelihood whose variance follows the Evans (2011) error model, learned jointly with the
    rest of the model:

        Var(I_obs) = a·σ² + b²·(s·Î)²

    σ is the integration error and s·Î the predicted intensity averaged over the Monte Carlo samples (s the scale, Î
    the surrogate posterior's intensity), detached by default. a and b are trained on the ELBO with everything else.

    No leave-one-out is needed. The ELBO averages the likelihood over the surrogate posterior, so each observation's
    expected squared residual is (I_obs − s·μ)² + s²·σ_q². The σ_q² term cancels the bias from the observation having
    pulled the posterior toward itself (the EM update for a variance). This holds only if σ_q is calibrated.
    """
    def __init__(self, a=1., b=0.01, degrees_of_freedom=None, detach=True):
        super().__init__()
        self.detach = detach
        self.log_a = torch.nn.Parameter(torch.tensor(math.log(a)))
        self.log_b = torch.nn.Parameter(torch.tensor(math.log(b)))
        self.degrees_of_freedom = degrees_of_freedom

    @property
    def a(self):
        return torch.exp(self.log_a)

    @property
    def b(self):
        return torch.exp(self.log_b)

    def variance(self, ipred, sigiobs):
        # ipred is (reflections, mc samples) and sigiobs is (reflections, 1). The proportional term uses the sample
        # mean of s·I. Detached (the default), the scales and posterior cannot move it to change their own variance.
        imean = (ipred.detach() if self.detach else ipred).mean(-1, keepdim=True)
        return self.a * torch.square(sigiobs) + torch.square(self.b * imean)

    def forward(self, ipred, iobs, sigiobs):
        scale = torch.sqrt(self.variance(ipred, sigiobs))
        if self.degrees_of_freedom is None:
            likelihood = td.Normal(iobs, scale)
        else:
            likelihood = td.StudentT(self.degrees_of_freedom, iobs, scale)
        return likelihood.log_prob(ipred)

    @torch.no_grad()
    def register_metrics(self, ipred, iobs, sigiobs):
        super().register_metrics(ipred, iobs, sigiobs)
        tracking.add_metric(self.a, 'Ev11a')
        tracking.add_metric(self.b, 'Ev11b')
