import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase

class TruncatedNormal(tfd.TruncatedNormal):
    def sample(self, *args, **kwargs):
        z = super().sample(*args, **kwargs)
        return tf.maximum(z, self.low)

class TruncatedNormalPosteriorBase(FoldedNormalPosteriorBase):
    """
    A base class for creating truncated normal posteriors. 
    """
    high = 1e10

    def _distribution(self, loc, scale, low):
        q = TruncatedNormal(
            loc + low, 
            scale, 
            low=low,
            high=self.high,
        )
        return q

