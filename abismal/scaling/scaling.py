import tensorflow as tf
from tensorflow_probability import distributions as tfd
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

def normal_posterior(output, bijector_function):
    loc, scale = tf.unstack(output, axis=-1)
    scale = bijector_function(scale)
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
    scale = scale + 1. #prevent change in concavity
    q = tfd.Gamma(loc, scale)
    return q

def delta_posterior(output, bijector_function):
    loc = tf.squeeze(output, axis=-1)
    q = DeltaDistribution(loc)
    return q

def elog(x):
    """ exponential-logarithmic constraint """
    neg = tf.minimum(x, 0.)
    pos = tf.maximum(x, 0.)
    return tf.math.exp(neg) + tf.math.log1p(pos)

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
    }
    prior_dict = {
        'cauchy' : lambda : tfd.Cauchy(0., 1.),
        'laplace' : lambda : tfd.Laplace(0., 1.),
        'normal' : lambda : tfd.Normal(0., 1.),
        'halfnormal' : lambda : tfd.HalfNormal(1.),
        'halfcauchy' : lambda : tfd.HalfCauchy(0., 1.),
        'exponential' : lambda : tfd.Exponential(1.),
        'lognormal' : lambda :  tfd.LogNormal(0., 1.),
    }
    def __init__(
            self, 
            mlp_width=32, 
            mlp_depth=20, 
            hidden_units=None,
            activation="relu",
            kl_weight=1.,
            epsilon=1e-12,
            num_image_samples=None,
            share_weights=True,
            prior_name='exponential',
            posterior_name='foldednormal',
            bijector_name='softplus',
            normalizer_name=None,
            hkl_to_imodel=False,
            gated=False,
            output_bias=True,
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
        normalizer_name : str (optional)
            The name of the normalizing function to use in the neural network
        hkl_to_imodel : bool (optional)
            Optionally allow the neural network to access the miller indices while computing the image representation vector. 
        gated : bool (optional)
            Optionally use a gated architecture instead of a vanilla residual multilayer perceptron
        output_bias : bool (optional)
            Use bias in the output layer. 
        """
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self.num_image_samples = num_image_samples
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.epsilon = epsilon
        self.activation = activation
        self.share_weights = share_weights
        self.prior_name = prior_name.lower()
        self.posterior_name = posterior_name.lower()
        self.bijector_name = bijector_name
        self.normalizer_name = normalizer_name
        self.hkl_to_imodel = hkl_to_imodel
        self.gated = gated
        self.output_bias = output_bias

        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = 2 * mlp_width 

        kernel_initializer = 'glorot_normal'
        self.input_image = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=False)
        self.input_scale = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=False) #Should use_bias?

        self.pool = Average(axis=-2)

        if gated:
            from abismal.layers import GLUFeedForward as FeedForward
        else:
            from abismal.layers import FeedForward 

        self.image_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units,
                    activation=self.activation,
                    kernel_initializer=kernel_initializer,
                    use_bias=False,
                    normalizer=normalizer_name,
                ) for _ in range(mlp_depth)])
        if share_weights:
            self.scale_network = self.image_network
        else:
            self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units, 
                    kernel_initializer=kernel_initializer, 
                    activation=self.activation, 
                    use_bias=False,
                    normalizer=normalizer_name,
                    ) for _ in range(mlp_depth)
            ]) 

        if self.posterior_name == 'delta':
            self.output_dense = tfk.layers.Dense(1, kernel_initializer=kernel_initializer, use_bias=output_bias)
        else:
            self.output_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=output_bias)

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
            'prior_name' : self.prior_name,
            'posterior_name' : self.posterior_name,
            'bijector_name' : self.bijector_name,
            'normalizer_name' : self.normalizer_name,
            'hkl_to_imodel' : self.hkl_to_imodel,
            'gated' : self.gated,
            'output_bias' : self.output_bias, 
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
        self.pool.build(metadata[:-1] + [self.mlp_width])
        self.output_dense.build(metadata[:-1] + [self.mlp_width])
        self.built = True

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
        if self.hkl_to_imodel:
            image.append(0.02 * tf.cast(hkl, metadata.dtype))


        image = tf.concat(image, axis=-1)

        if self.num_image_samples is not None:
            #Subsample reflections per image 
            image = ImageScaler.sample_refls(image, self.num_image_samples)

        image = self.input_image(image)
        image = tf.ragged.map_flat_values(self.image_network, image)
        image = self.pool(image)

        scale = tf.ragged.map_flat_values(self.input_scale, scale)
        scale = scale + image
        scale = tf.ragged.map_flat_values(self.scale_network, scale)
        scale = tf.ragged.map_flat_values(self.output_dense, scale)

        if ragged_tensor.is_ragged(scale):
            q = self.distribution_function(scale.flat_values)
        else:
            q = self.distribution_function(scale)

        z = q.sample(mc_samples) 

        if not self.posterior_name.lower() == 'delta' and self.kl_weight > 0.:
            p = self.prior_function()
            try: #Attempt to calculate this analytically
                kl_div = q.kl_divergence(p)
            except NotImplementedError:
                q_z = q.log_prob(z)
                z = z + self.epsilon 
                p_z = p.log_prob(z)
                kl_div = q_z - p_z
                #self.add_metric(tf.math.reduce_mean(q_z), name='Σ_q_z')
                #self.add_metric(tf.math.reduce_mean(p_z), name='Σ_p_z')

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

class KBImageScaler(ImageScaler):
    def build(self, shapes):
        super().build(shapes)
        self.B = self.add_weight(
            name='log_B', shape=(), dtype='float32', initializer='zeros'
        )
        self.log_k = self.add_weight(
            name='log_k', shape=(), dtype='float32', initializer='zeros'
        )

    def call(self, inputs, mc_samples=32, training=None, **kwargs):
        z = super().call(inputs, mc_samples, training, **kwargs)
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        k = tf.math.exp(self.log_k)
        self.add_metric(self.B, name='B')
        self.add_metric(k, name='k')
        z = tf.math.exp(self.log_k - self.B * tf.math.reciprocal(tf.math.square(resolution))) * z
        return z

