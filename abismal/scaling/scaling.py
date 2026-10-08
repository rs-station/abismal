import torch
import torch.nn.functional as F
from torch import distributions as td
from abismal import serialization
from abismal import tracking
from abismal.distributions import FoldedNormal,Rice
from abismal.layers.feed_forward import FeedForward, GLUFeedForward, linear, variance_scaling_

def _normalize(x, axis, epsilon=1e-3):
    out = (
        x - x.mean(axis, keepdim=True)
    ) / (
        x.std(axis, keepdim=True, correction=0) + epsilon
    )
    return out

def batch_normalize(x, epsilon=1e-3):
    return _normalize(x, 0, epsilon)

def instance_normalize(x, epsilon=1e-3):
    return _normalize(x, -2, epsilon)

def layer_normalize(x, epsilon=1e-3):
    return _normalize(x, -1, epsilon)

class DeltaDistribution():
    def __init__(self, loc):
        self.loc = loc

    def rsample(self, *args, **kwargs):
        """Return a tensor that broadcasts as a sample"""
        return self.loc[None,...]

    sample = rsample

def normal_posterior(output, bijector_function):
    loc, scale = output.unbind(-1)
    scale = bijector_function(scale)
    q = td.Normal(loc, scale)
    return q

def pos_normal_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = output.unbind(-1)
    q = td.Normal(loc, scale)
    return q

def log_normal_posterior(output, bijector_function):
    loc, scale = output.unbind(-1)
    scale = bijector_function(scale)
    q = td.LogNormal(loc, scale)
    return q

def folded_normal_posloc_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = output.unbind(-1)
    q = FoldedNormal(loc, scale)
    return q

def folded_normal_posterior(output, bijector_function):
    loc, scale = output.unbind(-1)
    scale = bijector_function(scale)
    q = FoldedNormal(loc, scale)
    return q

def rice_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = output.unbind(-1)
    q = Rice(loc, scale)
    return q

def gamma_posterior(output, bijector_function):
    output = bijector_function(output)
    loc, scale = output.unbind(-1)
    q = td.Gamma(loc, scale)
    return q

def delta_posterior(output, bijector_function):
    loc = output.squeeze(-1)
    q = DeltaDistribution(loc)
    return q

def elog(x):
    """ exponential-logarithmic constraint """
    neg = torch.clamp_max(x, 0.)
    pos = torch.clamp_min(x, 0.)
    return torch.exp(neg) + torch.log1p(pos)

# The priors below keep TFP's positional conventions, which are not always the
# obvious ones: `exponential_prior`'s "scale" is the rate, and `gamma_prior`
# passes (rate, conc) as (concentration, rate).
def cauchy_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.Cauchy(loc, scale)

def laplace_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.Laplace(loc, scale)

def normal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.Normal(loc, scale)

def cen_normal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.Normal(torch.zeros_like(scale), scale)

def cen_laplace_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.Laplace(torch.zeros_like(scale), scale)

def lognormal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.LogNormal(loc, scale)

def halfcauchy_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.HalfCauchy(scale)

def halfnormal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.HalfNormal(scale)

def exponential_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
    return td.Exponential(scale)

def foldednormal_prior(loc=0., scale=1., bijector_function=None):
    if bijector_function is not None:
        scale = bijector_function(scale)
        loc = bijector_function(loc)
    return FoldedNormal(loc, scale)

def gamma_prior(rate=1., conc=1., bijector_function=None):
    if bijector_function is not None:
        rate = bijector_function(rate)
        conc = bijector_function(conc)
    return td.Gamma(rate, conc)

def _has_analytic_kl(q_type, p_type):
    """Whether torch has a finite, closed form KL divergence for this pair."""
    try:
        fn = td.kl._dispatch_kl(q_type, p_type)
    except NotImplementedError:
        return False
    if fn is NotImplemented:
        return False
    return "infinity" not in fn.__name__

def kl_divergence(q, p, z, epsilon=0.):
    """
    The analytic KL divergence where torch has one, otherwise a Monte Carlo
    estimate from the samples `z` drawn from `q`.

    torch registers some pairs with mismatched support -- a normal posterior
    against an exponential prior, say -- as infinitely divergent. TFP has no
    such entries and abismal has always estimated those pairs by sampling, so
    a non-finite analytic value falls through to the estimate as well.
    """
    if _has_analytic_kl(type(q), type(p)):
        return td.kl_divergence(q, p)
    q_z = q.log_prob(z)
    p_z = p.log_prob(z + epsilon)
    return q_z - p_z

@serialization.register
class ImageScaler(torch.nn.Module, serialization.Serializable):
    bijector_dict = {
        'softplus' : F.softplus,
        'elup1' : lambda x : F.elu(x) + 1.,
        'exp' : torch.exp,
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
            metadata_length,
            mlp_width=32,
            mlp_depth=20,
            hidden_units=None,
            activation="relu",
            kl_weight=1.,
            epsilon=1e-12,
            ff_epsilon=None,
            share_weights=True,
            prior_name='exponential',
            posterior_name='foldednormal',
            bijector_name='softplus',
            pre_activation=None,
            hkl_to_imodel=False,
            gated=False,
            output_bias=True,
            optimize_prior_scale=False,
            dropout=None,
            random_seed=1234,
            batch_normalize=True,
            metadata_noise_factor=0.1,
        ):
        """
        This function has a lot of overrides, but comes with sensible defaults built in.

        Parameters
        ----------
        metadata_length : int
            The number of metadata columns per reflection.
        mlp_width : int (optional)
            Default 32 neurons. This is referred to as d-model in the CLI / paper
        mlp_depth : int (optional)
            Default 20 layers
        hidden_units : int (optional)
            This defaults to 2*mlp_width
        activation : str (optional)
            The name of an activation function with the default being "relu".
        kl_weight : float (optional)
            The importance of the prior distribution on scales. This parameter is ignored if the posterior is a delta distribution.
        epsilon : float (optional)
            A small constant for numerical stability defaults to 1e-12.
        ff_epsilon : float (optional)
            Epsilon used by the feed forward layers' `pre_activation`, when that is a
            normalizer. Defaults to None, in which case it falls back to `epsilon`.
        share_weights : bool (optional)
            Whether or not share neural network weights between the image model and the scale model.
            The default is True.
        prior_name : str (optional)
            The name of the prior distribution to use.
        posterior_name : str (optional)
            The posterior parameterization to use
        bijector_name : str (optional)
            The bijector to use for parameters that need to be constrained positive
        pre_activation : str (optional)
            Applied to each feed forward layer's input. Either the name of a normalizer
            (a key of `FeedForward.norm_dict`) or an activation name.
        hkl_to_imodel : bool (optional)
            Optionally allow the neural network to access the miller indices while computing the image representation vector.
        gated : bool (optional)
            Optionally use a gated architecture instead of a vanilla residual multilayer perceptron
        output_bias : bool (optional)
            Use bias in the output layer.
        dropout : float (optional)
            Use dropout when pooling reflections.
        random_seed : int (optional)
            Seeds the weight initialization.
        metadata_noise_factor : float (optional)
            The standard deviation of the Gaussian noise added to the metadata. This regularizes
            the model against overfitting on small data sets. The noise is only applied when
            `training` is True. The default is 0.1.
        """
        super().__init__()
        self.metadata_length = metadata_length
        self.kl_weight = kl_weight
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.epsilon = epsilon
        self.ff_epsilon = ff_epsilon
        self.activation = activation
        self.share_weights = share_weights
        self.prior_name = prior_name.lower()
        self.posterior_name = posterior_name.lower()
        self.bijector_name = bijector_name
        self.pre_activation = pre_activation
        self.hkl_to_imodel = hkl_to_imodel
        self.gated = gated
        self.output_bias = output_bias
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.optimize_prior_scale = optimize_prior_scale
        self.random_seed = random_seed
        self.batch_normalize = batch_normalize
        self.metadata_noise_factor = metadata_noise_factor

        input_bias=False

        if self.hidden_units is None:
            self.hidden_units = 2 * mlp_width
        ffepsilon = epsilon if ff_epsilon is None else ff_epsilon

        # Seed the initialization without disturbing the global stream that the
        # Monte Carlo sampling draws from.
        with torch.random.fork_rng(devices=[]):
            if random_seed is not None:
                torch.manual_seed(random_seed)
            kernel_initializer = lambda w: variance_scaling_(w, mlp_depth**-1.0, 'fan_avg') #FixUp init for early layers

            n = 2
            dimage = metadata_length + n
            if hkl_to_imodel:
                dimage = dimage + 3

            if optimize_prior_scale:
                self.output_prior = linear(mlp_width, 2, output_bias, kernel_initializer)

            self.input_image = linear(dimage, mlp_width, input_bias, kernel_initializer)
            self.input_scale = linear(metadata_length, mlp_width, input_bias, kernel_initializer) #Should use_bias?

            FF = GLUFeedForward if gated else FeedForward

            def network():
                return torch.nn.Sequential(*[
                    FF(
                        mlp_width,
                        hidden_units=self.hidden_units,
                        activation=self.activation,
                        kernel_initializer=kernel_initializer,
                        use_bias=False,
                        pre_activation=pre_activation,
                        epsilon=ffepsilon,
                    ) for _ in range(mlp_depth)])

            self.image_network = network()
            if share_weights:
                self.scale_network = self.image_network
            else:
                self.scale_network = network()

            if self.posterior_name == 'delta':
                self.output_dense = linear(mlp_width, 1, output_bias, kernel_initializer)
            else:
                self.output_dense = linear(mlp_width, 2, output_bias, kernel_initializer)

    def get_config(self):
        config = {
            'metadata_length' : self.metadata_length,
            'mlp_width' : self.mlp_width,
            'mlp_depth' : self.mlp_depth,
            'hidden_units': self.hidden_units,
            'epsilon' : self.epsilon,
            'ff_epsilon' : self.ff_epsilon,
            'activation' : self.activation,
            'kl_weight' : self.kl_weight,
            'share_weights' : self.share_weights,
            'prior_name' : self.prior_name,
            'posterior_name' : self.posterior_name,
            'bijector_name' : self.bijector_name,
            'pre_activation' : self.pre_activation,
            'hkl_to_imodel' : self.hkl_to_imodel,
            'gated' : self.gated,
            'output_bias' : self.output_bias,
            'optimize_prior_scale': self.optimize_prior_scale,
            'dropout': self.dropout,
            'random_seed': self.random_seed,
            'batch_normalize' : self.batch_normalize,
            'metadata_noise_factor' : self.metadata_noise_factor,
        }
        return config

    def bijector_function(self, x):
        return self.bijector_dict[self.bijector_name](x) + self.epsilon

    def distribution_function(self, output):
        return self.posterior_dict[self.posterior_name](output, self.bijector_function)

    @staticmethod
    def pool(image, image_id, num_images):
        """
        Leave-one-out mean of `image` over the reflections of each image.

        Parameters
        ----------
        image : torch.Tensor
            Per-reflection features, shape (reflections, channels)
        image_id : torch.Tensor
            Which image each reflection belongs to, shape (reflections,)
        num_images : int
        """
        sums = image.new_zeros((num_images, image.shape[-1])).index_add_(0, image_id, image)
        counts = torch.bincount(image_id, minlength=num_images).to(image.dtype)
        num = sums[image_id] - image
        den = (counts[image_id] - 1.).clamp_min(1.)
        return num / den[:, None]

    def forward(self, inputs, mc_samples=32, training=None):
        """
        Parameters
        ----------
        inputs : abismal.io.Batch
            Flat, per-reflection tensors and the image each reflection belongs to.
        mc_samples : int
        training : bool

        Returns
        -------
        z : torch.Tensor
            Samples of the scale of each reflection, shape (reflections, mc_samples)
        """
        hkl = inputs.hkl
        metadata = inputs.metadata
        iobs = inputs.iobs
        sigiobs = inputs.sigiobs
        image_id = inputs.image_id
        num_images = inputs.num_images

        if training and self.metadata_noise_factor > 0.:
            metadata = metadata + self.metadata_noise_factor * torch.randn_like(metadata)

        image = [metadata, iobs, sigiobs]
        if self.hkl_to_imodel:
            image.append(0.02 * hkl.to(metadata.dtype))

        image = torch.cat(image, dim=-1)
        if self.batch_normalize:
            image = batch_normalize(image)

        # The scale branch sees the (normalized) metadata columns only.
        scale = image[..., :self.metadata_length]

        unpooled_image = self.input_image(image)
        image = self.image_network(unpooled_image)
        image = self.pool(image, image_id, num_images)

        scale_in = self.input_scale(scale) + image
        q_latent = self.scale_network(scale_in)
        q_params = self.output_dense(q_latent)

        q = self.distribution_function(q_params)

        z = q.rsample((mc_samples,))

        if not self.posterior_name.lower() == 'delta' and self.kl_weight > 0.:
            if self.optimize_prior_scale:
                p_params = self.output_prior(image)
                m,s = p_params.unbind(-1)
                p = self.prior_dict[self.prior_name](m, s, self.bijector_function)
                tracking.add_metric(p.mean.mean(), 'pΣ_loc')
                tracking.add_metric(p.stddev.mean(), 'pΣ_scale')
            else:
                p = self.prior_dict[self.prior_name]()

            kl_div = kl_divergence(q, p, z, self.epsilon)
            kl_div = kl_div.mean()
            tracking.add_loss(self.kl_weight * kl_div)
            tracking.add_metric(kl_div, name='KL_Σ')

        z = z.transpose(0, -1)

        tracking.add_metric(z.mean(), name='Σ_mean')
        tracking.add_metric(z.std(correction=0), name='Σ_std')
        return z
