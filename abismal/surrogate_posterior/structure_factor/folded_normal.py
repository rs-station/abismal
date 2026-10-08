from abismal import serialization
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase


@serialization.register
class FoldedNormalPosterior(FoldedNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a folded normal distribution
    """


