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

    fit='online' estimates a and b outside the ELBO, in closed form, from running sums (online_update). Each training
    batch's observations add to the normal equations of the weighted least-squares fit of their squared residuals,

        y = (I_obs − p)²,  x = (σ², p²),  E[y] = a·σ² + b²·p²,  p the mean prediction over the Monte Carlo samples,

    with weights 1/v̂² at the current estimate v̂ (times the Student t's E-step weight (ν + 1)/(ν + y/v̂) if the
    likelihood is a t), rescaled so that every batch weighs as much as its number of observations. (a, b²) solves the accumulated 2×2 system, with both kept non-negative. The ELBO sees a and b
    as constants. Nothing accumulates for the first `burnin_steps` training steps, while the residuals still reflect an
    unfitted model; a and b hold their starting values until then. The sums restart after 10 and 100 accumulating
    steps, so the remembered fit is weighted at a good estimate. After that they are cumulative (each new batch moves
    the estimate by about 1/t), or decay by 1 − 1/decay_steps per step if decay_steps > 0.
    """
    def __init__(self, a=1., b=0.01, degrees_of_freedom=None, detach=True, fixed=False, fit='elbo', burnin_steps=0,
                 decay_steps=0, fit_b=True):
        super().__init__()
        if fit not in ('elbo', 'online'):
            raise ValueError(f"fit must be 'elbo' or 'online', not {fit!r}")
        self.detach = detach
        self.fit = fit
        # fixed: a and b stay at their given values (not trained, so the optimizer never sees them).
        trainable = fit == 'elbo' and not fixed
        # fit_b=False: b stays at its starting value whichever way a is fitted.
        self.log_a = torch.nn.Parameter(torch.tensor(math.log(a)), requires_grad=trainable)
        self.log_b = torch.nn.Parameter(torch.tensor(math.log(b)), requires_grad=trainable and fit_b)
        self.degrees_of_freedom = degrees_of_freedom
        if fit == 'online':
            self.burnin_steps = int(burnin_steps)
            self.fit_b = fit_b
            self.decay = 1. - 1. / decay_steps if decay_steps > 0 else 1.
            # The estimate (a, b²), the normal equations' running sums and the training step count.
            self.register_buffer('theta', torch.tensor([a, b * b], dtype=torch.float64))
            self.register_buffer('xx', torch.zeros(2, 2, dtype=torch.float64))
            self.register_buffer('xy', torch.zeros(2, dtype=torch.float64))
            self.register_buffer('steps', torch.zeros((), dtype=torch.int64))

    @property
    def a(self):
        if self.fit == 'online':
            return self.theta[0].float()
        return torch.exp(self.log_a)

    @property
    def b(self):
        if self.fit == 'online':
            return torch.sqrt(self.theta[1]).float()
        return torch.exp(self.log_b)

    @torch.compiler.disable
    @torch.no_grad()
    def online_update(self, ipred, iobs, sigiobs, valid):
        """
        fit='online': add one training batch to the running sums and re-solve for (a, b²). `valid` marks the
        observations that count (shape like iobs or (reflections,)). Runs eagerly: compiled, it pinned the batch size
        and recompiled the training step for every batch.
        """
        self.steps += 1
        active = (self.steps > self.burnin_steps).to(torch.float64)
        p = ipred.mean(-1).reshape(-1).double()
        y = torch.square(iobs.reshape(-1).double() - p)
        x = torch.stack([torch.square(sigiobs.reshape(-1).double()), torch.square(p)], -1)
        v = (x * self.theta).sum(-1).clamp_min(1e-30)
        w = valid.reshape(-1).to(torch.float64) / torch.square(v)
        if self.degrees_of_freedom is not None:
            nu = float(self.degrees_of_freedom)
            w = w * (nu + 1.) / (nu + y / v)
        # Each batch counts in proportion to its valid observations, whatever the scale of v̂: otherwise the first
        # batches, weighted at the starting a and b, can outweigh everything after them by orders of magnitude.
        n = valid.reshape(-1).to(torch.float64).sum()
        w = w * n / w.sum().clamp_min(1e-300) * active
        # Hold the (empty) sums during the burn-in. After 10 and after 100 accumulating steps, start them over: the
        # first batches are weighted at a poor v̂, and their leverage on b would otherwise outlast every later batch
        # (iteratively reweighted least squares, done twice).
        k = self.steps - self.burnin_steps
        restart = ((k == 11) | (k == 101)).to(torch.float64)
        keep = (self.decay * active + (1. - active)) * (1. - restart)
        self.xx.copy_(keep * self.xx + torch.einsum('n,ni,nj->ij', w, x, x))
        self.xy.copy_(keep * self.xy + torch.einsum('n,ni,n->i', w, x, y))
        # Solve the 2×2 system; if either coefficient comes out negative, fit the other alone with it at zero.
        (s11, s12), (_, s22) = self.xx
        y1, y2 = self.xy
        det = s11 * s22 - s12 * s12
        a = (s22 * y1 - s12 * y2) / det
        b2 = (s11 * y2 - s12 * y1) / det
        a_only = (y1 / s11).clamp_min(0.)
        b2_only = (y2 / s22).clamp_min(0.)
        both = (a > 0) & (b2 >= 0) & (det > 0)
        theta = torch.where(both, torch.stack([a, b2]),
                torch.where(a > 0, torch.stack([a_only, torch.zeros_like(a)]), torch.stack([torch.zeros_like(a), b2_only])))
        if not self.fit_b:
            # b held at its starting value: a alone fits what the b term leaves, a = Σw·σ²·(y − b²p²) / Σw·σ⁴.
            b2 = self.theta[1]
            theta = torch.stack([((y1 - b2 * s12) / s11).clamp_min(0.), b2])
        # Keep the starting values until the sums hold data.
        ready = (s11 > 0) & torch.isfinite(theta).all() & (theta[0] > 0)
        self.theta.copy_(torch.where(ready, theta, self.theta))

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
