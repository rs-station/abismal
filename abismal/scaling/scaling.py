import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from IPython import embed

from abismal_zero.layers import *
from abismal_zero.blocks import *
   
class ImageEmbeddingBlock(tfk.layers.Layer):
    def __init__(self, 
        dropout=0.0, 
        hidden_units=None,
        layer_norm=False,
        activation="ReLU",
        kernel_initializer='glorot_normal',
        reduce=False,
        **kwargs, 
        ):
        super().__init__(**kwargs)
        self.resnet = ResnetLayer(
            hidden_units=hidden_units, 
            normalize=layer_norm, 
            kernel_initializer=kernel_initializer, 
            activation=activation, 
        )
        self.embedding = SoftmaxEmbedding(kernel_initializer=kernel_initializer, dropout=dropout)
        self.reduce = reduce

    def call(self, data, **kwargs):
        out = self.resnet(data)
        if self.reduce:
            out = self.embedding(out)
        else:
            out = out + self.embedding(out)
        return out

class ImageScaler(tfk.layers.Layer):
    def __init__(self, 
        mlp_width, 
        mlp_depth, 
        dropout=0.0, 
        hidden_units=None,
        kl_weight=1., 
        prior=None,
        layer_norm=False,
        activation="ReLU",
        kernel_initializer='glorot_normal',
        eps=1e-6,
        stop_f_grad=True,
        **kwargs, 
        ):
        super().__init__(**kwargs)

        self.prior = prior
        self.kl_weight = kl_weight

        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        self.input_image   = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        self.input_scale  = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)

        self.image_network = tfk.models.Sequential([
                #ImageEmbeddingBlock(
                #    dropout=dropout, 
                #    hidden_units=hidden_units, 
                #    layer_norm=layer_norm, 
                #    activation=activation, 
                #    kernel_initializer=kernel_initializer,
                ResnetLayer(
                    dropout=dropout,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    normalize=layer_norm,
                ) for i in range(mlp_depth)
            ] + [
                ImageEmbeddingBlock(
                    dropout=dropout, 
                    hidden_units=hidden_units, 
                    layer_norm=layer_norm, 
                    activation=activation, 
                    kernel_initializer=kernel_initializer,
                    reduce=True,
                )
        ])

        self.scale_network = tfk.models.Sequential([
                ResnetLayer(
                    hidden_units=hidden_units, 
                    normalize=layer_norm, 
                    kernel_initializer=kernel_initializer, 
                    activation=activation, 
                    ) for i in range(mlp_depth)
            ] + [
                tfk.layers.Dense(2, kernel_initializer=kernel_initializer),
        ])

        self.stop_f_grad = stop_f_grad
        self.scale_bijector = tfb.Chain([tfb.Shift(1e-6), tfb.Exp()])
        self.loc_bijector = None

    def call(self, inputs, mc_samples=1, return_embeddings=False, **kwargs):
        metadata, iobs, sigiobs, imodel = inputs
        
        if self.stop_f_grad:
            imodel = tf.stop_gradient(imodel)

        scale = metadata
        image = tf.concat((metadata, iobs, sigiobs, imodel), axis=-1)

        image = self.input_image(image)
        scale = self.input_scale(scale)

        image = self.image_network(image)
        scale = scale + image

        params = self.scale_network(scale)
        loc, scale = tf.unstack(params.flat_values, axis=-1)
        if self.scale_bijector is not None:
            scale = self.scale_bijector(scale)
        if self.loc_bijector is not None:
            loc = self.loc_bijector(loc)

        q = tfd.Normal(loc, scale)
        z = q.sample(mc_samples)

        if self.prior is not None:
            try:
                kl_div = q.kl_divergence(self.prior)
                kl_div = tf.reduce_mean(kl_div)
            except NotImplementedError:
                kl_div = q.log_prob(z) - self.prior.log_prob(z)
                kl_div = tf.reduce_mean(kl_div)
            self.add_metric(kl_div, name="Scale KL")
            self.add_loss(self.kl_weight * kl_div)

        z = tf.RaggedTensor.from_row_splits(tf.transpose(z), params.row_splits)
        #z = tf.exp(z)
        return z

