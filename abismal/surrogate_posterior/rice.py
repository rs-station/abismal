from abismal.distributions import Rice
from abismal.surrogate_posterior._shift import Shifted
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase



#TODO: refactor with shared base class
class RicePosteriorBase(FoldedNormalPosteriorBase):
    """
    A base class for creating Rician posteriors. 
    """
    def _distribution(self, loc, scale, low):
        f = Rice(
            loc, 
            scale, 
        )
        q = Shifted(f, low)
        return q
