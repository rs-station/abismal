import math
import numpy as np
import tensorflow as tf
from abismal.layers import Standardize,FeedForward,Average,NormPool,ConvexCombination
from abismal.distributions import TruncatedNormal,FoldedNormal
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import stats as tfs
from tensorflow.python.ops.ragged import ragged_tensor
import tf_keras as tfk

from abismal.distributions import FoldedNormal,Rice


class DeltaDistribution():
    def __init__(self, loc):
        self.loc = loc

    def sample(self, *args, **kwargs):
        """Return a tensor that broadcasts as a sample"""
        return self.loc[None,...]

    def kl_divergence(self, *args, **kwargs):
        return 0.

@tfk.saving.register_keras_serializable(package="abismal")
class ImageScaler(tfk.models.Model):
    prior_dict = {
        #scale = scale + self.epsilon
        #output = tf.nn.softplus(output)
        #loc, scale = tf.unstack(output, axis=-1)
        #scale = scale + 0.001 * loc
        'cauchy' : lambda x: tfd.Cauchy(0., x),
        'laplace' : lambda x: tfd.Laplace(0., 0.5 * x),
        'normal' : lambda x: tfd.Normal(0., x),
        'halfnormal' : lambda x: tfd.HalfNormal(x/math.sqrt(1. - 2. / math.pi)),
        'halfcauchy' : lambda x: tfd.HalfCauchy(x),
        'exponential' : lambda x: tfd.Exponential(1. / x),
        'moyal' : lambda x: tfd.Exponential(math.sqrt(2.) * x / math.pi),
    }
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
            prior_name='cauchy',
            posterior_name='normal',
            seed=1234,
            normalize=None,
            skip=True,
            standardization_decay=0.999,
            standardization_epsilon=1e-3,
            hkl_to_imodel=False,
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
        prior_name : str (optional)
            The scale prior to use. See the self.prior_dict attribute for a list of current priors.
        posterior_name : str (optional)
            The posterior parameterization to use. Curently, normal, foldednormal, and gamma are supported.
            Normal is the default. The prior must have the same support as the posterior. 
        standardization_decay : float (optional)
            Sets the amount of memory for the online standardization of intensities and metadata.
        standardization_epsilon : float (optional)
            A small number to add to the denominator during standardization
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
        if not prior_name is None:
            prior_name = prior_name.lower()
        self.prior_name = prior_name
        self.posterior_name = posterior_name.lower()
        self.skip = skip
        self.standardization_decay = standardization_decay
        self.standardization_epsilon = standardization_epsilon
        self.hkl_to_imodel = hkl_to_imodel
        self.inflate_output = True


        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = 2 * image_mpl_width

        kernel_initializer = tfk.initializers.VarianceScaling(
            scale=1./10./mlp_depth,
            mode='fan_avg', distribution='truncated_normal', seed=seed
        )

        self.input_image = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=False)
        self.input_scale = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=True)

        self.pool = Average(axis=-2)

        ff_bias = False
        image_network = []
        for i in range(mlp_depth):
            image_network.append(
                FeedForward(
                    hidden_units=self.hidden_units,
                    activation=self.activation,
                    kernel_initializer=kernel_initializer,
                    normalize=normalize,
                    skip=skip,
                    use_bias=ff_bias,
                )
            )
        self.image_network = tfk.models.Sequential(image_network)

        if share_weights:
            self.scale_network = self.image_network
        else:
            self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units, 
                    kernel_initializer=kernel_initializer, 
                    activation=self.activation, 
                    normalize=normalize,
                    skip=skip,
                    use_bias=ff_bias,
                    ) for i in range(mlp_depth)
            ]) 

        if posterior_name == 'delta':
            self.output_dense = tfk.layers.Dense(1, kernel_initializer=kernel_initializer, use_bias=False)
        else:
            self.output_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=False)

        self.standardize_intensity = Standardize(
            center=False, decay=standardization_decay, epsilon=standardization_epsilon,
            #count_max=2_000,
        )
        self.standardize_metadata = Standardize(
            center=True, decay=standardization_decay, epsilon=standardization_epsilon,
            #count_max=2_000,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_width' : self.mlp_width,
            'mlp_depth' : self.mlp_depth, 
            'hidden_units': self.hidden_units,
            'activation' : self.activation,
            'kl_weight' : self.kl_weight,
            'epsilon' : self.epsilon,
            'num_image_samples' : self.num_image_samples,
            'share_weights' : self.share_weights,
            'prior_name' : self.prior_name,
            'posterior_name' : self.posterior_name,
            'seed' : self.seed,
            'normalize' : self.image_network.layers[0].normalize,
            'skip' : self.skip,
            'standardization_decay' : self.standardization_decay,
            'standardization_epsilon' : self.standardization_epsilon,
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

    def prior_function(self, scale):
        return self.prior_dict[self.prior_name](scale)

    def bijector_function(self, x):
        return tf.nn.softplus(x) + self.epsilon
        #return tf.math.exp(x) + self.epsilon

    def normal_posterior(self, output):
        loc, scale = tf.unstack(output, axis=-1)
        scale = self.bijector_function(scale)
        q = tfd.Normal(loc, scale)
        return q

    def log_normal_posterior(self, output):
        output = self.bijector_function(output)
        loc, scale = tf.unstack(output, axis=-1)
        #scale = self.bijector_function(scale)
        q = tfd.LogNormal(loc, scale)
        return q

    def truncated_normal_posterior(self, output):
        output = self.bijector_function(output)
        loc, scale = tf.unstack(output, axis=-1)
        q = tfd.TruncatedNormal(loc, scale, 0., math.inf)
        return q

    def folded_normal_posterior(self, output):
        #output = self.bijector_function(output)
        loc, scale = tf.unstack(output, axis=-1)
        scale = self.bijector_function(scale)
        q = FoldedNormal(loc, scale)
        return q

    def gamma_posterior(self, output):
        output = self.bijector_function(output)
        loc, scale = tf.unstack(output, axis=-1)
        q = tfd.Gamma(loc, scale)
        return q

    def delta_posterior(self, output):
        loc = tf.squeeze(output, axis=-1)
        q = DeltaDistribution(loc)
        return q

    def distribution_function(self, output):
        posterior_dict = {
            'normal' : self.normal_posterior,
            'gamma' : self.gamma_posterior,
            'foldednormal' : self.folded_normal_posterior,
            'truncatednormal' : self.truncated_normal_posterior,
            'delta' : self.delta_posterior,
            'lognormal' : self.log_normal_posterior,
        }
        return posterior_dict[self.posterior_name](output)

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
        if self.hkl_to_imodel:
            dimage += 3
        self.input_image.build(
            metadata[:-1] + [dimage] #add columns for iobs/sigiobs
        )

        self.input_scale.build(metadata)
        #self.image_network.build(metadata[:-1] + [self.mlp_width])
        self.image_network.build([None, self.mlp_width])
        if not self.share_weights:
            #self.scale_network.build(metadata[:-1] + [self.mlp_width])
            self.scale_network.build([None, self.mlp_width])
        self.pool.build(metadata[:-1] + [self.mlp_width])
        self.output_dense.build(metadata[:-1] + [self.mlp_width])

        self.standardize_intensity.build(iobs)
        self.standardize_metadata.build(metadata)

        self.built = True

    def standardize_inputs(self, inputs, training=None):
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        iobs = tf.ragged.map_flat_values(
            self.standardize_intensity,
            iobs,
            training=training,
        )
        sigiobs = tf.ragged.map_flat_values(
            self.standardize_intensity.standardize,
            sigiobs,
        )
        metadata = tf.ragged.map_flat_values(
            self.standardize_metadata,
            metadata,
            training=training,
        )
        out = (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        )
        self.add_metric(self.standardize_intensity.std, "Istd")
        return out

    def call(self, inputs, mc_samples=32, training=None, **kwargs):
        inputs = self.standardize_inputs(inputs, training=training)
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
        if self.hkl_to_imodel:
            image.append(tf.cast(hkl, 'float32') / 50.)

        image = tf.concat(image, axis=-1)

        if self.num_image_samples is not None:
            #Subsample reflections per image 
            image = ImageScaler.sample_refls(image, self.num_image_samples)

        image = tf.ragged.map_flat_values(self.input_image, image)
        scale = tf.ragged.map_flat_values(self.input_scale, scale)

        image = tf.ragged.map_flat_values(self.image_network, image)
        image = self.pool(image)

        self.add_metric(tf.math.reduce_mean(image), "Image")
        self.add_metric(tf.math.reduce_std(image), "ImageStd")
        scale = scale + image

        scale = tf.ragged.map_flat_values(self.scale_network, scale)
        scale = tf.ragged.map_flat_values(self.output_dense, scale)

        if ragged_tensor.is_ragged(scale):
            q = self.distribution_function(scale.flat_values)
        else:
            q = self.distribution_function(scale)

        z = q.sample(mc_samples) 


        if self.prior_name is not None:
            if self.inflate_output:
                p = self.prior_function(1.)
            else:
                p = self.prior_function(self.standardize_intensity.std)

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

        # Inflate to scale of intensities
        if self.inflate_output:
            z = self.standardize_intensity.std * z


        self.add_metric(tf.math.reduce_mean(z), name='Σ_mean')
        self.add_metric(tf.math.reduce_std(z), name='Σ_std')

        return z

