import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.truncated_normal import TruncatedNormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class TruncatedNormalPosterior(TruncatedNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a truncated normal distribution
    """
