import torch
from torch import distributions as td
from abismal import serialization
from abismal.prior.wilson import WilsonPriorBase,AutoWilsonPriorBase


def WilsonDistribution(centric, epsilon, sigma=1.):
    concentration = torch.where(centric, 0.5, 1.)
    rate = torch.where(centric, 0.5 / sigma / epsilon, 1. / sigma / epsilon)
    return td.Gamma(concentration, rate)

@serialization.register
class WilsonPrior(WilsonPriorBase):
    """Wilson's priors on intensities."""
    def distribution(self, asu_id=None, hkl=None):
        if asu_id is None:
            centric = self.rac.centric
            epsilon = self.rac.epsilon
            sigma = self.sigma
            p = WilsonDistribution(centric, epsilon, sigma)
            return p
        centric = self.rac.gather(self.rac.centric, asu_id, hkl)
        epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
        sigma = self.rac.gather(self.sigma, asu_id, hkl)
        p = WilsonDistribution(centric, epsilon, sigma)
        return p

    def flat_distribution(self):
        return self.distribution()

@serialization.register
class AutoWilsonPrior(WilsonPrior, AutoWilsonPriorBase):
    """Wilson prior with learnable scale and b-factor"""
