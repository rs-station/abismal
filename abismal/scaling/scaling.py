import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import stats as tfs
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
            bijector=None,
        ):
        super().__init__()
        self.dense = tfk.layers.Dense(1, kernel_initializer=kernel_initializer)
        self.bijector = None

    def call(self, data, training=None, **kwargs):
        theta = self.dense(data)
        theta = tf.squeeze(theta, axis=-1)
        if self.bijector is not None:
            theta = self.bijector(theta)
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

class OnlineMoments(tfk.layers.Layer):
    def __init__(self, axis=-1):
        self.axis = axis
        self.freeze = False

    def build(self, shape):
        shape = shape[self.axis]
        self.mean = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self.var = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='variance',
        )
        self.count = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.int32,
            trainable=False,
            name='count',
        )
        self.count = 0

    def call(self, data):
        if training:
            self.count += 1
            if not self.freeze:
                tfs.assign_moving_mean_variance(
                    data,
                    self.mean,
                    self.var,
                    axis=-1
                )
        return 

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
            num_image_samples=96,
            hkl_to_image_model=True,
            hkl_divisor=1.,
            metadata_model=None,
            **kwargs, 
        ):
        super().__init__(**kwargs)

        self.metadata_model = metadata_model
        self.hkl_to_image_model = hkl_to_image_model
        self.num_image_samples = num_image_samples
        self.hkl_divisor = hkl_divisor

        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        input_image   = [
            tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer),
        ]
        if dropout is not None:
            input_image.append(tfk.layers.Dropout(dropout))
        self.input_image   = tfk.models.Sequential(input_image)
        self.input_scale  = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        self.pool = Average(axis=-2)

        self.image_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    normalize=layer_norm,
                ) for i in range(mlp_depth)])

        self.scale_network = self.image_network
        #self.scale_network = tfk.models.Sequential([
        #    FeedForward(
        #        hidden_units=hidden_units, 
        #        normalize=layer_norm, 
        #        kernel_initializer=kernel_initializer, 
        #        activation=activation, 
        #        ) for i in range(mlp_depth)
        #]) 

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

    def call(self, inputs, imodel, mc_samples=1, training=None, **kwargs):
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs


        if self.stop_f_grad:
            imodel = tf.stop_gradient(imodel)

        if self.metadata_model is not None:
            metadata = tf.ragged.map_flat_values(self.metadata_model, metadata)
        scale = metadata
        if self.hkl_to_image_model:
            hkl_float = tf.cast(hkl, dtype='float32') / self.hkl_divisor
            image = tf.concat((hkl_float, metadata, iobs, sigiobs, imodel), axis=-1)
        else:
            image = tf.concat((metadata, iobs, sigiobs, imodel), axis=-1)
        image = ImageScaler.sample_ragged_dim(image, self.num_image_samples)

        image = self.input_image(image)
        scale = self.input_scale(scale)

        image = self.image_network(image)
        image = self.pool(image)
        scale = scale + image

        scale = self.scale_network(scale)
        q = self.scale_posterior(scale.flat_values, training=training)
        z = q.sample(mc_samples)
        z = tf.RaggedTensor.from_row_splits(tf.transpose(z), metadata.row_splits)
        return z

