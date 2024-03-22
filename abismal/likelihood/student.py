import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd

class StudentTLikelihood(tfk.layers.Layer):
    def __init__(self, degrees_of_freedom):
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom

    def call(self, ipred, iobs, sigiobs):
        return tfd.StudentT(self.degrees_of_freedom, iobs, sigiobs).log_prob(ipred)

