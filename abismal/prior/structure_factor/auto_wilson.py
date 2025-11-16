import numpy as np
import tensorflow as tf
from abismal.distributions import FoldedNormal,Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection
from abismal.prior.base import PriorBase
from abismal.prior.structure_factor.wilson import WilsonDistribution

@tfk.saving.register_keras_serializable(package="abismal")
class AutoWilsonPrior(PriorBase):
    """Auto scaling Wilson prior."""
    def __init__(self, rac, **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.log_B = self.add_weight(name='log_B', shape=(), dtype='float32', initializer='zeros')
        self.log_k = self.add_weight(name='log_k', shape=(), dtype='float32', initializer='zeros')

    @property
    def B(self):
        return tf.math.exp(self.log_B)

    @property
    def k(self):
        return tf.math.exp(self.log_k)

    @property
    def sigma(self):
        return tf.math.exp(self.log_k + -self.B * tf.math.rsqrt(self.rac.dHKL))

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)

    def _distribution(self, asu_id=None, hkl=None):
        sigma = self.sigma
        if asu_id is None:
            centric = self.rac.centric
            epsilon = self.rac.epsilon
            asu_id = self.rac.asu_id
        else:
            centric = self.rac.gather(self.rac.centric, asu_id, hkl)
            epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
            sigma = self.rac.gather(sigma, asu_id, hkl)
        p = WilsonDistribution(centric, epsilon, sigma)
        return p

    def flat_distribution(self):
        return self._distribution(asu_id=None, hkl=None)

    def distribution(self, asu_id, hkl):
        return self._distribution(asu_id, hkl)

    def call(self, asu_id=None, hkl=None, **kwargs):
        self.add_metric(self.B, name='B')
        self.add_metric(self.k, name='k')
        #self.add_loss(1e-3 * self.k * self.k)
        return super().call(asu_id, hkl)
