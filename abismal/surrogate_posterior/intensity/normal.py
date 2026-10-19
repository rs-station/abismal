from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase,MultivariateNormalPosteriorBase
class NormalPosterior(NormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a normal distribution
    """
class MultivariateNormalPosterior(MultivariateNormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a low-rank multivariate normal distribution
    """
