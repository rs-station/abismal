import math
import numpy as np
import tensorflow as tf
from abismal.layers import Standardize
from abismal.distributions import TruncatedNormal,FoldedNormal
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import stats as tfs
from tensorflow.python.ops.ragged import ragged_tensor
import tf_keras as tfk

from abismal.layers import *
from abismal.distributions import FoldedNormal,Rice


class DeltaDistribution():
    def __init__(self, loc):
        self.loc = loc

    def sample(self, *args, **kwargs):
        """Return a tensor that broadcasts as a sample"""
        return self.loc[None,...]


@tfk.saving.register_keras_serializable(package="abismal")
class ImageScaler(tfk.models.Model):
    def __init__(
            self, 
            mlp_width=32, 
            mlp_depth=20, 
            hidden_units=None,
            activation="LeakyReLU",
            kl_weight=1.,
            epsilon=1e-12,
            num_image_samples=None,
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
            This is a Keras activation function with the default being "leaky_relu"
        kl_weight : float (optional)
            The importance of the prior distribution on scales. If <=0, the model will use a delta distribution
            for the posterior and no kl divergence will be calculated. 
        epsilon : float (optional)
            A small constant for numerical stability defaults to 1e-12. 
        num_image_samples : int (optional)
            The number of reflections to sample in order to create the image representation vectors. 
            No subsampling will be done if this is set to None which is the default. 
        share_weghts : bool (optional)
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

        self.input_image = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=True)
        self.input_scale = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=True)

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

        if self.kl_weight > 0.:
            self.output_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer)
        else:
            self.output_dense = tfk.layers.Dense(1, kernel_initializer=kernel_initializer)

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

    @staticmethod
    def sample_refls(tensor, mc_samples):
        n = tf.shape(tensor)[0]
        l = tf.reduce_sum(tf.ones_like(tensor[...,0]), axis=-1)
        idx = tf.random.uniform(
            (n, mc_samples),
            minval=-0.5,
            maxval=l[:,None]-0.5,
        )
        idx = tf.cast(tf.round(idx), 'int32')
        out = tf.gather(tensor, idx, axis=1, batch_dims=1)
        return out

    def prior_function(self):
        #p = tfd.Exponential(1.)
        p = tfd.Laplace(0., 1.)
        return p

    def bijector_function(self, x):
        return tf.nn.softplus(x) + self.epsilon

    def distribution_function(self, output):
        if self.kl_weight > 0.:
            loc, scale = tf.unstack(output, axis=-1)
            scale = self.bijector_function(scale)
            q = tfd.Normal(loc, scale)

            self.add_metric(tf.reduce_mean(loc), name='Σ_loc')
            self.add_metric(tf.reduce_mean(scale), name='Σ_scale')
            return q
        loc = tf.squeeze(output, axis=-1)
        loc = self.bijector_function(loc)
        self.add_metric(tf.reduce_mean(loc), name='Σ_loc')
        q = DeltaDistribution(loc)
        return q

    def build(self, shapes):
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = shapes

        dimage = metadata[-1] + 2
        self.input_image.build(
            metadata[:-1] + [dimage] #add columns for iobs/sigiobs
        )

        self.input_scale.build(metadata)
        self.image_network.build(metadata[:-1] + [self.mlp_width])
        if not self.share_weights:
            self.scale_network.build(metadata[:-1] + [self.mlp_width])
        self.pool.build(metadata[:-1] + [self.mlp_width])
        self.output_dense.build(metadata[:-1] + [self.mlp_width])

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
        image = [metadata, iobs, sigiobs]

        image = tf.concat(image, axis=-1)

        if self.num_image_samples is not None:
            #Subsample reflections per image 
            image = ImageScaler.sample_refls(image, self.num_image_samples)

        image = self.input_image(image)

        scale = tf.ragged.map_flat_values(self.input_scale, scale)

        image = self.image_network(image)
        image = self.pool(image)
        scale = scale + image

        scale = tf.ragged.map_flat_values(self.scale_network, scale)
        scale = tf.ragged.map_flat_values(self.output_dense, scale)

        if ragged_tensor.is_ragged(scale):
            q = self.distribution_function(scale.flat_values)
        else:
            q = self.distribution_function(scale)

        z = q.sample(mc_samples) 

        if self.kl_weight > 0.:
            p = self.prior_function()
            try:
                kl_div = q.kl_divergence(p)
            except NotImplementedError:
                q_z = q.log_prob(z)
                p_z = p.log_prob(z)
                kl_div = q_z - p_z

            kl_div = tf.reduce_mean(kl_div)
            self.add_loss(self.kl_weight * kl_div)
            self.add_metric(kl_div, name='KL_Σ')

        if ragged_tensor.is_ragged(scale):
            z = tf.RaggedTensor.from_row_splits(
                tf.transpose(z), metadata.row_splits
            )
        else:
            z = tf.transpose(z)

        self.add_metric(tf.math.reduce_mean(z), name='Σ_mean')
        self.add_metric(tf.math.reduce_std(z), name='Σ_std')
        return z

