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
        if training:
            if self.prior is not None:
                kl_div=q.kl_divergence(self.prior)
                kl_div = tf.reduce_mean(kl_div)
                self.add_loss(kl_div)
                self.add_metric(kl_div, name='KL_Σ')
        return q

class ImageScaler(tfk.layers.Layer):
    def __init__(
            self, 
            mlp_width, 
            mlp_depth, 
            dropout=0.0, 
            hidden_units=None,
            layer_norm=False,
            activation="ReLU",
            kernel_initializer='glorot_normal',
            stop_f_grad=True,
            scale_posterior=None,
            kl_weight=1.,
            eps=1e-12,
            **kwargs, 
        ):
        super().__init__(**kwargs)


        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        self.input_image   = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        self.input_scale  = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)

        self.image_network = tfk.models.Sequential([
                FeedForward(
                    dropout=dropout,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    normalize=layer_norm,
                ) for i in range(mlp_depth)
            ] + [
                ConvexCombination(
                    kernel_initializer=kernel_initializer,
                    dropout=dropout
                )
        ])

        self.scale_network = tfk.models.Sequential([
            FeedForward(
                hidden_units=hidden_units, 
                normalize=layer_norm, 
                kernel_initializer=kernel_initializer, 
                activation=activation, 
                ) for i in range(mlp_depth)
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

    def call(self, inputs, mc_samples=1, training=None, **kwargs):
        metadata, iobs, sigiobs, imodel = inputs
        
        if self.stop_f_grad:
            imodel = tf.stop_gradient(imodel)

        scale = metadata
        image = tf.concat((metadata, iobs, sigiobs, imodel), axis=-1)

        image = self.input_image(image)
        scale = self.input_scale(scale)

        image = self.image_network(image)
        scale = scale + image

        scale = self.scale_network(scale)
        q = self.scale_posterior(scale.flat_values, training=training)
        z = q.sample(mc_samples)
        z = tf.RaggedTensor.from_row_splits(tf.transpose(z), metadata.row_splits)
        return z

