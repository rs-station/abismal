import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd

class NormalLikelihood(tfk.layers.Layer):
    def call(self, ipred, iobs, sigiobs):
        return tfd.Normal(iobs, sigiobs).log_prob(ipred)

