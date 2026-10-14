import torch
from abismal.distributions import FoldedNormal
from abismal.surrogate_posterior.surrogate_posterior import PositiveParameter
from abismal.surrogate_posterior._shift import Shifted


class FoldedNormalPosteriorBase(object):
    """
    A base class for creating folded normal posteriors. 
    """
    def __init__(self, rac, loc_init=None, scale_init=None, epsilon=1e-12, **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.low = self.epsilon

        if loc_init is None:
            loc_init = torch.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * torch.ones(rac.asu_size)

        self.loc = self.positive(loc_init)
        self.scale = self.positive(scale_init, shift=epsilon)

    def _distribution(self, loc, scale, low):
        f = FoldedNormal(
            loc, 
            scale, 
        )
        q = Shifted(f, low)
        return q

    def distribution(self, asu_id, hkl):
        loc = self.loc.gather(self.rac, asu_id, hkl)
        scale = self.scale.gather(self.rac, asu_id, hkl)
        q = self._distribution(loc, scale, self.low)
        return q

    def flat_distribution(self):
        q = self._distribution(self.loc(), self.scale(), self.low)
        return q
