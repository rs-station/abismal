from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.rice import RicePosteriorBase
import tf_keras as tfk


@tfk.saving.register_keras_serializable(package="abismal")
class RicePosterior(RicePosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a Rice distribution
    """

