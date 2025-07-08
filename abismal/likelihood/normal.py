import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import stats as tfs
import tf_keras as tfk
from abismal.likelihood.location_scale import LocationScale

class NormalLikelihood(LocationScale):
    def _likelihood(self, ipred, iobs, sigiobs):
        return tfd.Normal(iobs, sigiobs)

class LeastSquaresLikelihood(LocationScale):
    def _likelihood(self, ipred, iobs, sigiobs):
        scale = tf.reduce_mean(sigiobs)
        return tfd.Normal(iobs, scale)

class TukeyNormal():
    def __init__(self, loc, scale, percentile=0.05, **kwargs):
        self.loc = loc
        self.scale = scale
        self.percentile = percentile

    def log_prob(self, X):
        ll = tfd.Normal(self.loc, self.scale).log_prob(X)
        ell = tf.reduce_mean(ll, axis=0, keepdims=True)
        cutoff = tfs.percentile(ell, self.percentile)
        ll = tf.where(ell >= cutoff, ll, 0.)
        return ll

class TukeyNormalLikelihood(LocationScale):
    def _likelihood(self, ipred, iobs, sigiobs):
        return TukeyNormal(iobs, sigiobs)
