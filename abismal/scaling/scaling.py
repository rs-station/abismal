import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from IPython import embed

from abismal.layers import *
from abismal.distributions import FoldedNormal

class DeltaDist():
    def __init__(self, vals):
        self.vals=vals

    def sample(self, *args, **kwargs):
        return self.vals[None,...]

class DeltaFunctionLayer(tfk.layers.Layer):
    def __init__(
            self, 
            kernel_initializer='glorot_normal',
        ):
        super().__init__()
        self.dense = tfk.layers.Dense(1, kernel_initializer=kernel_initializer)

    def call(self, data, training=None, **kwargs):
        theta = self.dense(data)
        theta = tf.squeeze(theta, axis=-1)
        q = DeltaDist(theta)
        return q

class LogNormalLayer(tfk.layers.Layer):
    def __init__(
            self, 
            kl_weight=1.,
            prior=None,
            eps=1e-12,
            kernel_initializer='glorot_normal',
        ):
        super().__init__()
        self.kl_weight=kl_weight
        self.prior=prior
        self.dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)
        self.eps = eps

    def call(self, data, training=None, **kwargs):
        theta = self.dense(data)
        loc, scale = tf.unstack(theta, axis=-1)
        scale = tf.math.exp(scale) + self.eps
        q = tfd.LogNormal(loc, scale)
        if self.prior is not None:
            kl_div=q.kl_divergence(self.prior)
            kl_div = tf.reduce_mean(kl_div)
            self.add_loss(self.kl_weight * kl_div)
            self.add_metric(kl_div, name='KL_Σ')
        return q

class NormalLayer(tfk.layers.Layer):
    def __init__(
            self, 
            kl_weight=1.,
            prior=None,
            eps=1e-12,
            kernel_initializer='glorot_normal',
        ):
        super().__init__()
        self.kl_weight=kl_weight
        self.prior=prior
        self.dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)
        self.eps = eps

    def call(self, data, training=None, **kwargs):
        theta = self.dense(data)
        loc, scale = tf.unstack(theta, axis=-1)
        scale = tf.math.exp(scale) + self.eps
        q = tfd.Normal(loc, scale)
        if self.prior is not None:
            kl_div=q.kl_divergence(self.prior)
            kl_div = tf.reduce_mean(kl_div)
            self.add_loss(self.kl_weight * kl_div)
            self.add_metric(kl_div, name='KL_Σ')
        return q

class TruncatedNormalLayer(tfk.layers.Layer):
    def __init__(
            self, 
            kl_weight=1.,
            prior=None,
            eps=1e-12,
            kernel_initializer='glorot_normal',
            low=0.,
            high=1e32
        ):
        super().__init__()
        self.low = low
        self.high = high
        self.kl_weight=kl_weight
        self.prior=prior
        self.dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)
        self.eps = eps

    def call(self, data, training=None, mc_samples=None, **kwargs):
        theta = self.dense(data)
        theta = tf.math.exp(theta) + self.eps
        loc, scale = tf.unstack(theta, axis=-1)
        q = tfd.TruncatedNormal(loc, scale, self.low, self.high)
        if mc_samples is not None:
            z = q.sample(mc_samples)
        else:
            z = q

        if self.prior is not None:
            if mc_samples is not None:
                kl_div = q.log_prob(z) - self.prior.log_prob(z)
            else:
                kl_div=q.kl_divergence(self.prior)
            kl_div = tf.reduce_mean(kl_div)
            self.add_loss(self.kl_weight * kl_div)
            self.add_metric(kl_div, name='KL_Σ')
        return z

class FoldedNormalLayer(tfk.layers.Layer):
    def __init__(
            self, 
            kl_weight=1.,
            prior=None,
            eps=1e-12,
            kernel_initializer='glorot_normal',
        ):
        super().__init__()
        self.kl_weight=kl_weight
        self.prior=prior
        self.dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)
        self.eps = eps

    def call(self, data, training=None, **kwargs):
        theta = self.dense(data)
        theta = tf.math.exp(theta) + self.eps
        loc, scale = tf.unstack(theta, axis=-1)
        q = FoldedNormal(loc, scale)
        if self.prior is not None:
            kl_div=q.kl_divergence(self.prior)
            kl_div = tf.reduce_mean(kl_div)
            self.add_loss(self.kl_weight * kl_div)
            self.add_metric(kl_div, name='KL_Σ')
        return q

class ImageScaler(tfk.layers.Layer):
    def __init__(
            self, 
            mlp_width, 
            mlp_depth, 
            ff_dropout=None, 
            cc_dropout=None, 
            hidden_units=None,
            layer_norm=False,
            activation="ReLU",
            kernel_initializer='glorot_normal',
            stop_f_grad=True,
            imodel=True,
            scale_posterior=None,
            kl_weight=1.,
            eps=1e-12,
            image_dims=None,
            **kwargs, 
        ):
        super().__init__(**kwargs)

        self.image_dims = image_dims

        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        self.input_image   = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        if self.image_dims is None:
            self.input_scale  = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        else:
            self.input_scale  = tfk.layers.Dense(mlp_width - self.image_dims, kernel_initializer=kernel_initializer)

        layers = [
                FeedForward(
                    dropout=ff_dropout,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    normalize=layer_norm,
                ) for i in range(mlp_depth)
            ] + [
                ConvexCombination(
                    kernel_initializer=kernel_initializer,
                    dropout=cc_dropout,
                )
        ]
        if self.image_dims is not None:
                layers.append(
                    tfk.layers.Dense(image_dims, kernel_initializer=kernel_initializer)
                )

        self.image_network = tfk.models.Sequential(layers)
        self.scale_network = tfk.models.Sequential([
            FeedForward(
                dropout=ff_dropout,
                hidden_units=hidden_units, 
                normalize=layer_norm, 
                kernel_initializer=kernel_initializer, 
                activation=activation, 
                ) for i in range(mlp_depth)
        ]) 
        L = 5
        self.xy_positional_encoding = tfk.models.Sequential([
            Normalization(),
            PositionalEncoding(L)
        ])
        self.hkl_positional_encoding = tfk.models.Sequential([
            Normalization(),
            PositionalEncoding(L)
        ])

        if scale_posterior is None:
            prior = FoldedNormal(0., np.sqrt(np.pi/2).astype('float32'))
            self.scale_posterior = FoldedNormalLayer(
                kl_weight=kl_weight,
                prior=prior,
                kernel_initializer=kernel_initializer,
                eps=eps,
            )
        else:
            self.scale_posterior = scale_posterior

        self.stop_f_grad = stop_f_grad
        self.imodel = True

    def call(self, inputs, mc_samples=1, training=None, **kwargs):
        hkl, xy, iobs, sigiobs, imodel = inputs
        hkl = tf.cast(hkl, dtype=tf.float32)

        xy = tf.ragged.map_flat_values(self.xy_positional_encoding, xy)
        hkl = tf.ragged.map_flat_values(self.hkl_positional_encoding, hkl)

        if self.stop_f_grad:
            imodel = tf.stop_gradient(imodel)

        scale = xy
        if self.imodel:
            image = tf.concat((hkl, xy, iobs, sigiobs, imodel), axis=-1)
        else:
            image = tf.concat((hkl, xy, iobs, sigiobs), axis=-1)

        image = self.input_image(image)
        scale = self.input_scale(scale)

        image = self.image_network(image) 
        if self.image_dims is not None:
            image = image * tf.ones_like(scale[...,:1])
            scale = tf.concat((scale, image), axis=-1)
        else:
            scale = scale + image

        scale = self.scale_network(scale)
        q = self.scale_posterior(scale.flat_values, training=training)
        #z = self.scale_posterior(scale.flat_values, training=training, mc_samples=mc_samples)
        z = q.sample(mc_samples)
        z = tf.RaggedTensor.from_row_splits(tf.transpose(z), xy.row_splits)
        return z

