"""
Likelihoods with a calibrated error model.

The reported uncertainties are inflated as in Evans (2011),

    sigma'^2 = SdFac^2 (sigma^2 + SdB mu + (SdAdd mu)^2)

where mu is the model's expected intensity of the observation, its mean over
the Monte Carlo samples. mu is detached: the structure factors and scales
cannot lower the objective by moving mu to shrink the variance.

The parameters are either learned with the rest of the model (`learn=True`,
"Ev11 joint"), or set from outside, by `abismal.callbacks.Mm24Calibrator`,
which fits them to pairwise differences of scaled, equivalent observations as
in Mittan-Moreau et al. (2025).
"""
import math
import torch
import torch.nn.functional as F
from torch import distributions as td
from abismal import serialization
from abismal import tracking
from abismal.likelihood.location_scale import LocationScale


def _softplus_inverse(x):
    return x + math.log(-math.expm1(-x))


@serialization.register
class CalibratedLikelihood(LocationScale):
    def __init__(self, degrees_of_freedom=None, learn=False, sdfac=1., sdb=0., sdadd=0., floor=1e-6):
        """
        Parameters
        ----------
        degrees_of_freedom : float (optional)
            A t error model with this many degrees of freedom; normal if None.
        learn : bool
            Learn SdFac, SdB and SdAdd by gradient descent with the model.
        sdfac, sdb, sdadd : float
            Initial (or fixed) values.
        floor : float
            Learned SdB and SdAdd are parameterized as floor + softplus(x).
        """
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom
        self.learn = learn
        self.floor = floor
        self._init = dict(sdfac=sdfac, sdb=sdb, sdadd=sdadd)
        if learn:
            self.log_sdfac = torch.nn.Parameter(torch.tensor(math.log(sdfac)))
            self.raw_sdb = torch.nn.Parameter(torch.tensor(_softplus_inverse(max(sdb, 2 * floor) - floor)))
            self.raw_sdadd = torch.nn.Parameter(torch.tensor(_softplus_inverse(max(sdadd, 2 * floor) - floor)))
        else:
            self.register_buffer("_sdfac", torch.tensor(float(sdfac)))
            self.register_buffer("_sdb", torch.tensor(float(sdb)))
            self.register_buffer("_sdadd", torch.tensor(float(sdadd)))

    def get_config(self):
        return {
            'degrees_of_freedom': self.degrees_of_freedom,
            'learn': self.learn,
            'floor': self.floor,
            **{k: float(v) for k, v in self.parameters_dict().items()},
        }

    def parameters_dict(self):
        if self.learn:
            return {
                'sdfac': self.log_sdfac.exp(),
                'sdb': self.floor + F.softplus(self.raw_sdb),
                'sdadd': self.floor + F.softplus(self.raw_sdadd),
            }
        return {'sdfac': self._sdfac, 'sdb': self._sdb, 'sdadd': self._sdadd}

    @torch.no_grad()
    def set_parameters(self, sdfac=None, sdb=None, sdadd=None):
        """Set the fixed parameters (`learn=False` only)."""
        if self.learn:
            raise ValueError("set_parameters is for a likelihood with fixed parameters")
        for name, value in (('_sdfac', sdfac), ('_sdb', sdb), ('_sdadd', sdadd)):
            if value is not None:
                getattr(self, name).fill_(float(value))

    def variance(self, sigiobs, mu):
        p = self.parameters_dict()
        return p['sdfac'] ** 2 * (sigiobs ** 2 + p['sdb'] * mu + (p['sdadd'] * mu) ** 2)

    def forward(self, ipred, iobs, sigiobs):
        mu = ipred.detach().mean(-1, keepdim=True).clamp_min(0.)
        scale = self.variance(sigiobs, mu).sqrt()
        if self.degrees_of_freedom is None:
            likelihood = td.Normal(iobs, scale)
        else:
            likelihood = td.StudentT(self.degrees_of_freedom, iobs, scale)
        for k, v in self.parameters_dict().items():
            tracking.add_metric(v, {'sdfac': 'SdFac', 'sdb': 'SdB', 'sdadd': 'SdAdd'}[k])
        return likelihood.log_prob(ipred)
