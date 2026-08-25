import tensorflow as tf
from tensorflow_probability import distributions as tfd
from abismal.likelihood.location_scale import LocationScale


class WeightedLeastSquaresLikelihood(LocationScale):
    def call(self, ipred, iobs, sigiobs):
        return tf.square(
            (ipred - iobs) / sigiobs
        )

class NormalLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        return tfd.Normal(iobs, sigiobs)

class LeastSquaresLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        scale = tf.reduce_mean(sigiobs)
        return tfd.Normal(iobs, scale)

