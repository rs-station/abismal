import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class NormalPosterior(NormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a normal distribution
    """


