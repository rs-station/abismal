from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase
class FoldedNormalPosterior(FoldedNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a folded normal distribution
    """


