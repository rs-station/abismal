from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from abismal.distributions.rice import Rice
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
        q = tfd.TransformedDistribution(
            f, 
            tfb.Shift(low),
        )
        return q


