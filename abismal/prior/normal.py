import numpy as np
import tensorflow as tf
from abismal.distributions import FoldedNormal,Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection
from abismal.prior.base import PriorBase

@tfk.saving.register_keras_serializable(package="abismal")
class NormalPrior(PriorBase):
    """Normally distributed prior."""
    def __init__(self, rac, loc=0., scale=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        loc : float (optional)
        scale : float (optional)
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.loc = loc
        self.scale = scale
        self.built = True #This is always true

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'loc' : self.loc,
            'scale' : self.scale,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)

    def flat_distribution(self):
        loc = self.loc * tf.ones(self.rac.asu_size)
        scale = self.scale * tf.ones(self.rac.asu_size)
        return tfd.Normal(loc, scale)

    def distribution(self, asu_id, hkl):
        ones = tf.ones_like(
            tf.squeeze(asu_id, axis=-1),
            dtype='float32',
        )
        loc = self.loc * ones
        scale = self.scale * ones
        return tfd.Normal(loc, scale)

@tfk.saving.register_keras_serializable(package="abismal")
class MultivariateNormalPrior(NormalPrior):
    """Multivariate normal distributed prior."""
    def flat_distribution(self):
        loc = self.loc * tf.ones(self.rac.asu_size)
        scale = self.scale * tf.ones(self.rac.asu_size)
        return tfd.MultivariateNormalDiag(loc, scale)

@tfk.saving.register_keras_serializable(package="abismal")
class HalfNormalPrior(PriorBase):
    """HalfNormally distributed prior."""
    def __init__(self, rac, scale=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        scale : float (optional)
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.scale = scale
        self.built = True #This is always true

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'scale' : self.scale,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)

    def flat_distribution(self):
        scale = self.scale * tf.ones(self.rac.asu_size)
        return tfd.HalfNormal(scale)

    def distribution(self, asu_id, hkl):
        ones = tf.ones_like(
            tf.squeeze(asu_id, axis=-1),
            dtype='float32',
        )
        scale = self.scale * ones
        return tfd.HalfNormal(scale)

