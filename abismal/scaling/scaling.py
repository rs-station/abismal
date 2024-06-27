import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs
from abismal.distributions import TruncatedNormal,FoldedNormal
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import stats as tfs
import tf_keras as tfk

from abismal.layers import *
from abismal.distributions import FoldedNormal

class LocationScaleLayer(tfk.layers.Layer):
    def __init__(
            self, 
            eps=1e-12,
            kernel_initializer='glorot_normal',
        ):
        super().__init__()
        self.dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)
        self.eps = eps
        self.scale_bijector = tfb.Chain([tfb.Shift(eps), tfb.Exp()])

class FoldedNormalLayer(LocationScaleLayer):
    def call(self, data, training=None, mc_samples=None, **kwargs):
        theta = self.dense(data)
        loc, scale = tf.unstack(theta, axis=-1)

        if self.scale_bijector is not None:
            scale = self.scale_bijector(scale)

        q = FoldedNormal(loc, scale)
        return q

class ImageScaler(tfk.layers.Layer):
    def __init__(
            self, 
            mlp_width=32, 
            mlp_depth=20, 
            hidden_units=None,
            activation="ReLU",
            kernel_initializer=None,
            scale_posterior=None,
            scale_prior=None,
            kl_weight=1.,
            eps=1e-12,
            num_image_samples=32,
            share_weights=True,
            seed=None,
            **kwargs, 
        ):
        super().__init__(**kwargs)

        self.kl_weight = kl_weight
        self.num_image_samples = num_image_samples
        self.scale_prior = scale_prior
        self.scale_posterior = scale_posterior

        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        if kernel_initializer is None:
            if seed is None:
                seed = 1234 #This needs to be set to something otherwise all layers will
                #initialize the same
            kernel_initializer = tfk.initializers.VarianceScaling(
                scale=1./10./mlp_depth,
                mode='fan_avg', distribution='truncated_normal', seed=seed
            )

        input_image   = [
            tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer),
        ]
        input_scale = [
            tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer),
        ]
            
        self.input_image   = tfk.models.Sequential(input_image)
        self.input_scale   = tfk.models.Sequential(input_scale)

        self.pool = Average(axis=-2)

        self.image_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                ) for i in range(mlp_depth)])

        if share_weights:
            self.scale_network = self.image_network
        else:
            self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=hidden_units, 
                    kernel_initializer=kernel_initializer, 
                    activation=activation, 
                    ) for i in range(mlp_depth)
            ]) 

        if self.scale_prior is None and kl_weight > 0.:
            self.scale_prior = tfd.Exponential(1.)

        if scale_posterior is None:
            self.scale_posterior = FoldedNormalLayer(
                kernel_initializer=kernel_initializer,
                eps=eps,
            )


    @staticmethod
    def sample_ragged_dim(ragged, mc_samples):
        """
        Randomly subsample "length" entries from ragged with replacement.
        """
        l = ragged.row_lengths()
        n = tf.shape(ragged)[0]
        r = tf.random.uniform((n, mc_samples))
        idx2 = tf.round(r * tf.cast(l, 'float32')[:,None]- 0.5)
        idx2 = tf.cast(idx2, 'int32')
        idx1 = tf.range(n)[:,None] * tf.ones_like(idx2)
        idx = tf.stack((idx1, idx2), axis=-1)
        out = tf.gather_nd(ragged, idx)
        return out

    def call(self, inputs, mc_samples=32, training=None, **kwargs):
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        scale = metadata
        image = [iobs, sigiobs, metadata]

        image = tf.concat(image, axis=-1)
        image = ImageScaler.sample_ragged_dim(image, self.num_image_samples)

        image = self.input_image(image)
        scale = self.input_scale(scale)

        image = self.image_network(image)
        image = self.pool(image)
        scale = scale + image

        scale = self.scale_network(scale)
        q = self.scale_posterior(scale.flat_values, training=training)
        z = q.sample(mc_samples)

        if self.scale_prior is not None:
            try:
                kl_div = q.kl_divergence(self.scale_prior)
            except NotImplementedError:
                q_z = q.log_prob(z)
                p_z = self.scale_prior.log_prob(z)
                kl_div = q_z - p_z

            kl_div = tf.reduce_mean(kl_div)
            self.add_loss(self.kl_weight * kl_div)
            self.add_metric(kl_div, name='KL_Σ')
        z = tf.RaggedTensor.from_row_splits(tf.transpose(z), metadata.row_splits)

        return z
