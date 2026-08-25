import tensorflow as tf
from tensorflow_probability import distributions as tfd
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

