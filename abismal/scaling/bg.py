import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow.python.ops.ragged import ragged_tensor
import tf_keras as tfk
from abismal.layers import *
from abismal.distributions import FoldedNormal,Rice

def _normalize(x, axis, epsilon=1e-3):
    out = (
        x - tf.reduce_mean(x, axis=axis, keepdims=True)
    ) / (
        tf.math.reduce_std(x, axis=axis, keepdims=True) + epsilon
    )
    return out

def batch_normalize(x, epsilon=1e-3):
    return _normalize(x, -3, epsilon)

def instance_normalize(x, epsilon=1e-3):
    return _normalize(x, -2, epsilon)

def layer_normalize(x, epsilon=1e-3):
    return _normalize(x, -1, epsilon)

class DeltaDistribution():
    def __init__(self, loc):
        self.loc = loc

    def sample(self, *args, **kwargs):
        """Return a tensor that broadcasts as a sample"""
        return self.loc[None,...]

def normal_posterior(output, bijector_function):
    loc, scale = tf.unstack(output, axis=-1)
    scale = bijector_function(scale)
    q = tfd.Normal(loc, scale)
    return q

def pos_normal_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = tf.unstack(output, axis=-1)
    q = tfd.Normal(loc, scale)
    return q

def log_normal_posterior(output, bijector_function):
    loc, scale = tf.unstack(output, axis=-1)
    scale = bijector_function(scale)
    q = tfd.LogNormal(loc, scale)
    return q

def folded_normal_posloc_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = tf.unstack(output, axis=-1)
    #scale = bijector_function(scale)
    q = FoldedNormal(loc, scale)
    return q

def folded_normal_posterior(output, bijector_function):
    loc, scale = tf.unstack(output, axis=-1)
    scale = bijector_function(scale)
    q = FoldedNormal(loc, scale)
    return q

def rice_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = tf.unstack(output, axis=-1)
    q = Rice(loc, scale)
    return q

def gamma_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = tf.unstack(output, axis=-1)
    #scale = scale + 1. #prevent change in concavity
    q = tfd.Gamma(loc, scale)
    return q

def delta_posterior(output, bijector_function):
    loc, _ = tf.unstack(output, axis=-1)
    q = DeltaDistribution(loc)
    return q

def elog(x):
    """ exponential-logarithmic constraint """
    neg = tf.minimum(x, 0.)
    pos = tf.maximum(x, 0.)
    return tf.math.exp(neg) + tf.math.log1p(pos)

def rescale_distribution(p, mean=1.):
    return tfd.TransformedDistribution(
        p,
        tfb.Scale(1. / p.mean())
    )

def cauchy_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.Cauchy(loc, scale)

def laplace_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.Laplace(loc, scale)

def normal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.Normal(loc, scale)

def cen_normal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.Normal(0.0, scale)

def cen_laplace_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.Laplace(0.0, scale)

def lognormal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
        #loc = bijector_function(loc)
    p = tfd.LogNormal(loc, scale)
    #prior_scale = tfb.Scale(1. / p.stddev())
    #p = tfd.TransformedDistribution(
    #    p,
    #    prior_scale,
    #)
    return p

def halfcauchy_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.HalfCauchy(scale)

def halfnormal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.HalfNormal(scale)

def exponential_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return tfd.Exponential(scale)

def foldednormal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
        loc = bijector_function(loc)
    return FoldedNormal(loc, scale)

def gamma_prior(rate=1., conc=1., bijector_function=None):
    if bijector_function is not None:
        rate = bijector_function(rate)
        conc = bijector_function(conc)
    return tfd.Gamma(rate, conc)

@tfk.saving.register_keras_serializable(package="abismal")
class ImageScaler(tfk.models.Model):
    bijector_dict = {
        'softplus' : tf.nn.softplus,
        'elup1' : lambda x : tf.nn.elu(x) + 1.,
        'exp' : tf.math.exp,
        'elog' : elog,
    }
    posterior_dict = {
        'normal' : normal_posterior,
        'gamma' : gamma_posterior,
        'foldednormal' : folded_normal_posloc_posterior,
        'rice' : rice_posterior,
        'lognormal' : log_normal_posterior,
        'delta' : delta_posterior,
        'posnormal' : pos_normal_posterior,
    }
    prior_dict = {
        'cauchy' : cauchy_prior,
        'laplace' : laplace_prior,
        'normal' : normal_prior,
        'cennormal' : cen_normal_prior,
        'cenlaplace' : cen_laplace_prior,
        'halfnormal' : halfnormal_prior,
        'halfcauchy' : halfcauchy_prior,
        'exponential' : exponential_prior, 
        'lognormal' : lognormal_prior, 
        'foldednormal' : foldednormal_prior,
        'gamma' : gamma_prior,
    }
    def __init__(
            self, 
            mlp_width=32, 
            mlp_depth=20, 
            hidden_units=None,
            activation="relu",
            kl_weight=1.,
            epsilon=1e-12,
            ff_epsilon=None,
            num_image_samples=None,
            share_weights=True,
            prior_name='exponential',
            posterior_name='foldednormal',
            bijector_name='softplus',
            pre_activation=None,
            hkl_to_imodel=False,
            gated=False,
            output_bias=True,
            dropout=None,
            random_seed=1234,
            batch_normalize=True,
            **kwargs, 
        ):
        """
        This function has a lot of overrides, but comes with sensible defaults built in. 

        Parameters
        ----------
        mlp_width : int (optional)
            Default 32 neurons. This is referred to as d-model in the CLI / paper
        mlp_depth : int (optional)
            Default 20 layers
        hidden_units : int (optional)
            This defaults to 2*mlp_width
        activation : str (optional)
            This is a Keras activation function with the default being "relu". 
        kl_weight : float (optional)
            The importance of the prior distribution on scales. This parameter is ignored if the posterior is a delta distribution. 
        epsilon : float (optional)
            A small constant for numerical stability defaults to 1e-12.
        ff_epsilon : float (optional)
            Epsilon used in the feed forward layers' pre-normalization (see `pre_activation`).
            Defaults to None, in which case it falls back to `epsilon`.
        num_image_samples : int (optional)
            The number of reflections to sample in order to create the image representation vectors. 
            No subsampling will be done if this is set to None which is the default. 
        share_weights : bool (optional)
            Whether or not share neural network weights between the image model and the scale model. 
            The default is True. 
        prior_name : str (optional)
            The name of the prior distribution to use. 
        posterior_name : str (optional)
            The posterior parameterization to use
        bijector_name : str (optional)
            The bijector to use for parameters that need to be constrained positive
        hkl_to_imodel : bool (optional)
            Optionally allow the neural network to access the miller indices while computing the image representation vector. 
        gated : bool (optional)
            Optionally use a gated architecture instead of a vanilla residual multilayer perceptron
        output_bias : bool (optional)
            Use bias in the output layer. 
        dropout : float (optional)
            Use dropout when pooling reflections. 
        """
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self.num_image_samples = num_image_samples
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.epsilon = epsilon
        self.ff_epsilon = ff_epsilon
        self.activation = activation
        self.share_weights = share_weights
        self.prior_name = prior_name.lower()
        self.posterior_name = posterior_name.lower()
        self.bijector_name = bijector_name
        self.pre_activation= pre_activation
        self.hkl_to_imodel = hkl_to_imodel
        self.gated = gated
        self.output_bias = output_bias
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.random_seed = random_seed
        self.batch_normalize = batch_normalize
        self.noise_std = 0.0
        self.bg_posterior_name = 'normal'
        self.bg_prior_name = 'laplace'

        input_bias=False

        if self.hidden_units is None:
            self.hidden_units = 2 * mlp_width
        ffepsilon = epsilon if ff_epsilon is None else ff_epsilon

        #kernel_initializer = 'glorot_normal'
        kernel_initializer = tfk.initializers.VarianceScaling(scale=mlp_depth**-1.0, mode='fan_avg', seed=random_seed) #FixUp init for early layers

        self.input_image = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=input_bias)
        self.input_scale = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=input_bias) #Should use_bias?

        #self.pool = Average(axis=-2, dropout=dropout)
        #self.pool = ConvexCombination()

        if gated:
            from abismal.layers import GLUFeedForward as FeedForward
        else:
            from abismal.layers import FeedForward 

        self.image_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units,
                    activation=self.activation,
                    kernel_initializer=kernel_initializer,
                    use_bias=True,
                    pre_activation=pre_activation,
                    epsilon=ffepsilon,
                ) for _ in range(mlp_depth)])
        if share_weights:
            self.scale_network = self.image_network
        else:
            self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units, 
                    kernel_initializer=kernel_initializer, 
                    activation=self.activation, 
                    use_bias=True,
                    pre_activation=pre_activation,
                    epsilon=ffepsilon,
                    ) for _ in range(mlp_depth)
            ]) 

        self.output_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=output_bias)
        #self.bg_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=output_bias)
        #self.output_prior = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=output_bias)
        #self.input_prior = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer, use_bias=False)
        #self.built = True
        #self.prior_loc = self.add_weight('prior_loc', shape=(), initializer='zeros')
        #self.prior_scale = self.add_weight('prior_scale', shape=(), initializer='ones')

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_width' : self.mlp_width,
            'mlp_depth' : self.mlp_depth, 
            'hidden_units': self.hidden_units,
            'epsilon' : self.epsilon,
            'ff_epsilon' : self.ff_epsilon,
            'activation' : self.activation,
            'kl_weight' : self.kl_weight,
            'num_image_samples' : self.num_image_samples,
            'share_weights' : self.share_weights,
            'prior_name' : self.prior_name,
            'posterior_name' : self.posterior_name,
            'bijector_name' : self.bijector_name,
            'pre_activation' : self.pre_activation,
            'hkl_to_imodel' : self.hkl_to_imodel,
            'gated' : self.gated,
            'output_bias' : self.output_bias, 
            'dropout': self.dropout,
            'random_seed': self.random_seed,
            'batch_normalize' : self.batch_normalize,
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
        return self.prior_dict[self.prior_name]()

    def bijector_function(self, x):
        return self.bijector_dict[self.bijector_name](x) + self.epsilon

    def distribution_function(self, output):
        return self.posterior_dict[self.posterior_name](output, self.bijector_function)

    def bg_distribution_function(self, output):
        return self.posterior_dict[self.bg_posterior_name](output, self.bijector_function)

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
            dimage = dimage + 3
        self.input_image.build(
            metadata[:-1] + [dimage] #add columns for iobs/sigiobs
        )

        self.input_scale.build(metadata)
        self.image_network.build(metadata[:-1] + [self.mlp_width])
        if not self.share_weights:
            self.scale_network.build(metadata[:-1] + [self.mlp_width])
        #self.pool.build(metadata[:-1] + [self.mlp_width])
        #self.output_dense.build(metadata[:-1] + [2 * self.mlp_width])
        self.output_dense.build(metadata[:-1] + [self.mlp_width])
        self.bg_dense.build(metadata[:-1] + [self.mlp_width])
        #self.output_prior.build(metadata[:-1] + [self.mlp_width])
        #self.input_prior.build(metadata[:-1] + [1])
        self.built = True

    def compute_kl_terms(self, q, p, z, name='KL_Σ', reduction='mean'):
        if isinstance(q, DeltaDistribution):
            return 
        try: #Attempt to calculate this analytically
            kl_terms = q.kl_divergence(p)
        except NotImplementedError:
            q_z = q.log_prob(z)
            z = z + self.epsilon 
            #p_z = p.log_prob(z * p.mean() / tf.math.reduce_mean(z, axis=0, keepdims=True))
            p_z = p.log_prob(z)
            kl_terms = q_z - p_z

        if reduction == 'mean':
            kl_div = tf.reduce_mean(kl_terms)
        elif reduction == 'sum':
            kl_div = tf.reduce_mean(kl_terms)
        else:
            kl_div = reduction(kl_terms)

        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(kl_div, name=name)
        return kl_terms

    def pool(self, image):
        if self.num_image_samples is None:
            #return tf.math.reduce_mean(image, axis=-2, keepdims=True)
            num = tf.math.reduce_sum(image, axis=-2, keepdims=True) - image
            den = tf.cast(image.row_lengths(), 'float32') - 1.
            den = tf.maximum(den, 1.)
            out = num / den[:,None,None]
        else:
            out = tf.math.reduce_mean(image, axis=-2, keepdims=True)
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

        if self.noise_std > 0.:
            noise = tf.random.normal(
                tfk.backend.shape(metadata.flat_values),
            )
            metadata = metadata.with_flat_values(
                metadata.flat_values + self.noise_std * noise
            )

        image = [metadata, iobs, sigiobs]
        n = 2
        if self.hkl_to_imodel:
            image.append(0.02 * tf.cast(hkl, metadata.dtype))
            n = n + 3

        image = tf.concat(image, axis=-1)

        if self.batch_normalize:
            #image = tf.ragged.map_flat_values(instance_normalize, image)
            image = instance_normalize(image)
        metadata = image[...,:-n]

        scale = metadata

        if self.num_image_samples is not None:
            #Subsample reflections per image 
            image = ImageScaler.sample_refls(image, self.num_image_samples)

        unpooled_image = self.input_image(image)
        if self.dropout is not None:
            unpooled_image = tf.nn.dropout(unpooled_image, self.dropout)

        unpooled_image = tf.ragged.map_flat_values(self.image_network, unpooled_image)

        image = self.pool(unpooled_image)
        image = tf.ragged.map_flat_values(self.scale_network, image)

        scale_in = tf.ragged.map_flat_values(self.input_scale, scale)
        if self.dropout is not None:
            scale_in = tf.nn.dropout(scale_in, self.dropout)

        scale_in = scale_in + image
        #scale_in = tf.ragged.map_flat_values(self.input_scale, scale)
        #q_latent = tf.ragged.map_flat_values(self.scale_network, scale_in) + image
        q_latent = tf.ragged.map_flat_values(self.scale_network, scale_in)
        #q_latent = tf.concat((
        #        tf.ragged.map_flat_values(self.scale_network, scale_in), 
        #    image
        #), axis=-1)
        q_params = tf.ragged.map_flat_values(self.output_dense, q_latent)
        bg_params = tf.ragged.map_flat_values(self.bg_dense, q_latent)

        q = self.distribution_function(q_params.flat_values)
        qb = self.bg_distribution_function(bg_params.flat_values)

        z = q.sample(mc_samples) 
        b = qb.sample(mc_samples)

        if self.kl_weight > 0.:
            p = self.prior_function()
            kl_terms = self.compute_kl_terms(q, p, z, name='KL_Σ')

            pb = self.prior_dict[self.bg_prior_name]()
            kl_terms = self.compute_kl_terms(qb, pb, b, name='KL_BG')

        z = tf.RaggedTensor.from_row_splits(
            tf.transpose(z), metadata.row_splits
        )

        b = tf.RaggedTensor.from_row_splits(
            tf.transpose(b), metadata.row_splits
        )

        self.add_metric(tf.math.reduce_mean(z), name='Σ_mean')
        self.add_metric(tf.math.reduce_std(z), name='Σ_std')

        self.add_metric(tf.math.reduce_mean(b), name='BG_mean')
        self.add_metric(tf.math.reduce_std(b), name='GB_std')
        return z, b


