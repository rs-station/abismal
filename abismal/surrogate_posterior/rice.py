import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.distributions.rice import Rice
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase



#TODO: refactor with shared base class
class RicePosteriorBase(FoldedNormalPosteriorBase):
    """
    A base class for creating Rician posteriors. 
    """
    def _distribution(self, loc, scale, low):
        f = Rice(
            loc, 
            scale, 
        )
        q = tfd.TransformedDistribution(
            f, 
            tfb.Shift(low),
        )
        return q


