from abismal import serialization
from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase


@serialization.register
class FoldedNormalPosterior(FoldedNormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a folded normal distribution
    """
