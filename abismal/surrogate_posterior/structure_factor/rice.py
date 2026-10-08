from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.rice import RicePosteriorBase
from abismal import serialization


@serialization.register
class RicePosterior(RicePosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a Rice distribution
    """

