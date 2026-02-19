import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from abismal.likelihood.location_scale import LocationScale,Ev11Likelihood

class NormalLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        return tfd.Normal(iobs, sigiobs)

class LeastSquaresLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        scale = tf.reduce_mean(sigiobs)
        return tfd.Normal(iobs, scale)


class Ev11NormalLikelihood(Ev11Likelihood):
    def _likelihood(self, iobs, sigiobs):
        return tfd.Normal(iobs, sigiobs)

