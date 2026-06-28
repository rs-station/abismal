import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from abismal.prior.base import PriorBase
from abismal.prior.wilson import WilsonPriorBase,AutoWilsonPriorBase
import tf_keras as tfk


def WilsonDistribution(centric, epsilon, sigma=1.):
    concentration = tf.where(centric, 0.5, 1.)
    rate = tf.where(centric, 0.5 / sigma / epsilon, 1. / sigma / epsilon)
    return tfd.Gamma(concentration, rate)

class WilsonPrior(WilsonPriorBase):
    """Wilson's priors on intensities."""
    def distribution(self, asu_id=None, hkl=None):
        if asu_id is None:
            centric = self.rac.centric
            epsilon = self.rac.epsilon
            sigma = self.sigma
            p = WilsonDistribution(centric, epsilon, sigma)
            return p
        centric = self.rac.gather(self.rac.centric, asu_id, hkl)
        epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
        sigma = self.sigma
        if len(tf.shape(sigma)) > 0:
            sigma = self.rac.gather(sigma, asu_id, hkl)
        p = WilsonDistribution(centric, epsilon, sigma)
        return p

    def flat_distribution(self):
        return self.distribution()

class AutoWilsonPrior(WilsonPrior, AutoWilsonPriorBase):
    """Wilson prior with learnable scale and b-factor"""
