import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from IPython import embed

from abismal.layers import *
from abismal.distributions import FoldedNormal

class GraphImageScaler(tfk.layers.Layer):
    def __init__(self, 
        mlp_width, 
        mlp_depth, 
        dropout=0.0, 
        hidden_units=None,
        layer_norm=False,
        activation="ReLU",
        kernel_initializer='glorot_normal',
        stop_f_grad=True,
        num_heads=8,
        eps=1e-12,
        kl_weight=1.,
        scale_posterior=None,
        **kwargs, 
        ):
        super().__init__(**kwargs)


        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        self.input_image   = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        self.input_scale  = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)

        self.image_network = TransformerAverage(
            mlp_depth,
            num_heads=num_heads, key_dim=hidden_units,
            dropout=dropout,
            hidden_units=hidden_units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            normalize=layer_norm,
        )

        self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=hidden_units, 
                    normalize=layer_norm, 
                    kernel_initializer=kernel_initializer, 
                    activation=activation, 
                    ) for i in range(mlp_depth)
            ] + [
                tfk.layers.Dense(1, kernel_initializer=kernel_initializer),
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

