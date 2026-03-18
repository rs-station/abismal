import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase,MultivariateNormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class NormalPosterior(NormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a normal distribution
    """


@tfk.saving.register_keras_serializable(package="abismal")
class MultivariateNormalPosterior(MultivariateNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a low-rank multivariate normal distribution
    """

