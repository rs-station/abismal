import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from abismal.prior.base import PriorBase
import tf_keras as tfk


def WilsonDistribution(centric, epsilon, sigma=1.):
    concentration = tf.where(centric, 0.5, 1.)
    rate = tf.where(centric, 0.5 / sigma / epsilon, 1. / sigma / epsilon)
    return tfd.Gamma(concentration, rate)

class WilsonPrior(PriorBase):
    """Wilson's priors on intensities."""
    def __init__(self, rac, sigma=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. If this is an array it must be length rac.asu_size
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.sigma = sigma
        self.built = True #This is always true

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'sigma' : self.sigma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)

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

