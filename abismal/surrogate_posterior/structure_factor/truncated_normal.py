from abismal import serialization
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.truncated_normal import TruncatedNormalPosteriorBase


@serialization.register
class TruncatedNormalPosterior(TruncatedNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a truncated normal distribution
    """
