import torch
from torch import distributions as td
from abismal import serialization
from abismal.prior.base import PriorBase

@serialization.register
class NormalPrior(PriorBase):
    """Normally distributed prior."""
    def __init__(self, rac, loc=0., scale=1.):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        loc : float (optional)
        scale : float (optional)
        """
        super().__init__(rac)
        self.loc = loc
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'loc' : self.loc,
            'scale' : self.scale,
        })
        return config

    def _ones(self, shape):
        return torch.ones(shape, device=self.rac.dHKL.device)

    def flat_distribution(self):
        loc = self.loc * self._ones(self.rac.asu_size)
        scale = self.scale * self._ones(self.rac.asu_size)
        return td.Normal(loc, scale)

    def distribution(self, asu_id, hkl):
        ones = self._ones(asu_id.shape[:-1])
        loc = self.loc * ones
        scale = self.scale * ones
        return td.Normal(loc, scale)

@serialization.register
class MultivariateNormalPrior(NormalPrior):
    """Multivariate normal distributed prior."""
    def flat_distribution(self):
        loc = self.loc * self._ones(self.rac.asu_size)
        scale = self.scale * self._ones(self.rac.asu_size)
        # A diagonal normal over the whole ASU, written as a rank-1 low-rank
        # normal with a zero factor: the dense MultivariateNormal would need an
        # (asu_size, asu_size) covariance, and this pairs with the low-rank
        # posterior in torch's analytic KL registry.
        factor = torch.zeros(self.rac.asu_size, 1, device=loc.device)
        return td.LowRankMultivariateNormal(loc, factor, scale * scale)

@serialization.register
class HalfNormalPrior(PriorBase):
    """HalfNormally distributed prior."""
    def __init__(self, rac, scale=1.):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        scale : float (optional)
        """
        super().__init__(rac)
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale' : self.scale,
        })
        return config

    def flat_distribution(self):
        scale = self.scale * torch.ones(self.rac.asu_size, device=self.rac.dHKL.device)
        return td.HalfNormal(scale)

    def distribution(self, asu_id, hkl):
        ones = torch.ones(asu_id.shape[:-1], device=self.rac.dHKL.device)
        scale = self.scale * ones
        return td.HalfNormal(scale)
