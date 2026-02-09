import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import stats as tfs
from abismal.likelihood.location_scale import LocationScale
from abismal.distributions.folded_normal import FoldedNormal

class NormalLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs, *args, **kwargs):
        return tfd.Normal(iobs, sigiobs)

class LeastSquaresDistribution(tfd.Normal):
    def log_prob(self, x):
        return self.unnormalized_log_prob(x)

class LeastSquaresLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs, *args, **kwargs):
        return LeastSquaresDistribution(iobs, sigiobs)

class TrimmedNormalDistribution():
    def __init__(self, loc, scale, outlier_frac=0.01):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.outlier_frac = outlier_frac

    def log_prob(self, x):
        ll = tfd.Normal(self.loc, self.scale).log_prob(x)
        ll_ave = tf.math.reduce_sum(x, axis=-1)
        cutoff = tfs.percentile(ll_ave, 100. * self.outlier_frac)
        ll = tf.where(ll_ave[:,None] >= cutoff, ll, 0.)
        return ll

class TrimmedNormalLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs, *args, **kwargs):
        return TrimmedNormalDistribution(iobs, sigiobs)

