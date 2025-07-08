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
            'loc' : tfk.saving.serialize_keras_object(self.loc),
            'scale' : tfk.saving.serialize_keras_object(self.scale),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        config['loc'] = tfk.saving.deserialize_keras_object(config['loc'])
        config['scale'] = tfk.saving.deserialize_keras_object(config['scale'])
        return cls(**config)

    def flat_distribution(self):
        loc = self.loc 
        scale = self.scale 
        return tfd.Normal(loc, scale)


@tfk.saving.register_keras_serializable(package="abismal")
class MultivariateNormalPrior(NormalPrior):
    """Multivariate normal distributed prior."""
    def flat_distribution(self):
        loc = self.loc * tf.ones(self.rac.asu_size)
        scale = self.scale * tf.ones(self.rac.asu_size)
        return tfd.MultivariateNormalDiag(loc, scale)

