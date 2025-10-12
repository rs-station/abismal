import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase



class TruncatedNormalPosteriorBase(FoldedNormalPosteriorBase):
    """
    A base class for creating truncated normal posteriors. 
    """
    high = 1e32

    def _distribution(self, loc, scale, low):
        q = tfd.TruncatedNormal(
            loc, 
            scale, 
            low=low,
            high=self.high,
        )
        return q

