import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase,TruncatedNormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class FoldedNormalPosterior(FoldedNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a folded normal distribution
    """



@tfk.saving.register_keras_serializable(package="abismal")
class TruncatedNormalPosterior(TruncatedNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a truncated normal distribution
    """


