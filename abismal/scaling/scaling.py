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

@tfk.saving.register_keras_serializable(package="abismal")
class ImageScaler(tfk.models.Model):
    def __init__(
            self, 
            mlp_width=32, 
            mlp_depth=20, 
            hidden_units=None,
            activation="ReLU",
            kl_weight=1.,
            epsilon=1e-12,
            num_image_samples=32,
            share_weights=True,
            seed=1234,
            **kwargs, 
        ):
        """
        This function has a lot of overrides, but comes with sensible defaults built in. 

        Parameters
        ----------
        mlp_width : int (optional)
            Default 32 neurons
        mlp_depth : int (optional)
            Default 20 layers
        hidden_units : int (optional)
            This defaults to 2*mlp_width
        activation : str (optional)
            This is a Keras activation function with the default being "ReLU"
        kl_weight : float (optional)
            This toggles the behavior of the scale_posterior. If scale_posterior isn't passed, and
            kl_weight is 0., a Delta distribution is used. Otherwise, the default surrogate
            posterior is a FoldedNormal. 
        epsilon : float (optional)
            A small constant for numerical stability defaults to 1e-12. 
        num_image_samples : int (optional)
            The number of reflections to sample in order to create the image representation vectors. 
            The default is 32 samples. 
        share_weights : bool (optional)
            Whether or not share neural network weights between the image model and the scale model. 
            The default is True. 
        seed : int (optional)
            An int or tf random seed for initialization. 
        """
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self.num_image_samples = num_image_samples
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.epsilon = epsilon
        self.activation = activation
        self.share_weights = share_weights
        self.seed = seed

        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = 2 * image_mpl_width

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
                    hidden_units=self.hidden_units,
                    activation=self.activation,
                    kernel_initializer=kernel_initializer,
                ) for i in range(mlp_depth)])

        if share_weights:
            self.scale_network = self.image_network
        else:
            self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units, 
                    kernel_initializer=kernel_initializer, 
                    activation=self.activation, 
                    ) for i in range(mlp_depth)
            ]) 

        self.output_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_width' : self.mlp_width,
            'mlp_depth' : self.mlp_depth, 
            'hidden_units': self.hidden_units,
            'epsilon' : self.epsilon,
            'activation' : self.activation,
            'kl_weight' : self.kl_weight,
            'num_image_samples' : self.num_image_samples,
            'share_weights' : self.share_weights,
            'seed' : self.seed,
        })
        return config

    def distribution_function(self, output):
        loc, scale = tf.unstack(output, axis=-1)
        loc = tf.exp(loc) + self.epsilon
        scale = tf.exp(scale) + self.epsilon
        q = FoldedNormal(loc, scale)
        return q

    def scale_prior(self):
        return tfd.Exponential(1.)

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
        scale = self.output_dense(scale)

        q = self.distribution_function(scale.flat_values)
        z = q.sample(mc_samples)
        p = self.scale_prior()

        try:
            kl_div = q.kl_divergence(p)
        except NotImplementedError:
            q_z = q.log_prob(z)
            p_z = p.log_prob(z)
            kl_div = q_z - p_z

        kl_div = tf.reduce_mean(kl_div)
        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(kl_div, name='KL_Σ')
        z = tf.RaggedTensor.from_row_splits(tf.transpose(z), metadata.row_splits)

        return z
