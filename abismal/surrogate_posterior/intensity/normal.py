import tf_keras as tfk
from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase,MultivariateNormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class NormalPosterior(NormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a normal distribution
    """


@tfk.saving.register_keras_serializable(package="abismal")
class MultivariateNormalPosterior(MultivariateNormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a low-rank multivariate normal distribution
    """
