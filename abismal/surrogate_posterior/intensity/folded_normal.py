import tf_keras as tfk
from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class FoldedNormalPosterior(FoldedNormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a folded normal distribution
    """
