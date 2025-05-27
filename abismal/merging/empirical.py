import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gemmi
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from abismal.symmetry import Op
import tf_keras as tfk
from abismal.layers import Standardize


@tfk.saving.register_keras_serializable(package="abismal")
class EmpiricalMergingModel(tfk.models.Model):
    """
    A model which averages intensities online using Welford's Algorithm
    """
    def __init__(
        self,
        rac,
        scale_model=None,
        decay=0.999, 
        posterior_type='intensity',
    ):
        super().__init__()
        self.rac = rac
        self.decay = decay
        self.posterior_type = posterior_type
        self.scale_model = scale_model

    def build(self, shapes):
        if self.built:
            return
        if self.scale_model is not None:
            self.scale_model.build(shapes)

        self.mean = self.add_weight(
            shape=self.rac.asu_size,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self.var = self.add_weight(
            shape=self.rac.asu_size,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='variance',
        )
        self.built = True

    def get_config(self):
        config = super().get_config()
        for k in ['scale_model', 'rac']:
            if config[k] is not None:
                config[k] = tfk.saving.serialize_keras_object(config[k])
        return config

    @classmethod
    def from_config(cls, config):
        for k in ['scale_model', 'rac']:
            config[k] = tfk.saving.deserialize_keras_object(config[k])
        return cls(**config)

    def call(self, inputs, mc_samples=None, training=None, **kwargs):
        inputs = map(lambda x: x.flat_values, inputs)
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        iobs,sigiobs = tf.squeeze(iobs),tf.squeeze(sigiobs)
        if self.posterior_type is 'structure_factor':
            J = tfd.TruncatedNormal(iobs, sigiobs, 0., np.inf)
            z = J.sample(32)
            iobs = tf.reduce_mean(tf.sqrt(z), axis=0)

        miller_ids = self.rac._miller_ids(asu_id, hkl_in)

        mean = tf.gather(self.mean, miller_ids)
        variance = tf.gather(self.var, miller_ids)

        delta = iobs - mean

        d = self.decay
        mean = d * mean + (1. - d) * delta
        variance = d * variance + (1. - d) * delta * (iobs - mean)

        mean_update = tf.scatter_nd(
            miller_ids[:,None], 
            (1. - d) * delta, 
            (self.rac.asu_size,)
        )
        var_update = tf.scatter_nd(
            miller_ids[:,None], 
            (d - 1.) * variance + (1. - d) * delta * (iobs - mean),
            (self.rac.asu_size,)
        )

        self.mean.assign_add(mean_update)
        self.var.assign_add(var_update)
        return iobs
