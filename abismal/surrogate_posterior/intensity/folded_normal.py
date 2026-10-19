from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase
class FoldedNormalPosterior(FoldedNormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a folded normal distribution
    """
