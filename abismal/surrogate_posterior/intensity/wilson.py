import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def WilsonPrior(centric, epsilon, sigma=1.):
    concentration = tf.where(centric, 0.5, 1.)
    rate = tf.where(centric, 0.5 / sigma / epsilon, 1. / sigma / epsilon)
    return tfd.Gamma(concentration, rate)

