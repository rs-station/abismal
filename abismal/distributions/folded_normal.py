import tensorflow as tf
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
import numpy as np
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import absolute_value as abs_bijector
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers
from tensorflow_probability import math as tfm


def reparameterization_gradient(z, loc, scale):
    a = (z - loc) / scale
    b = (z + loc) / scale

    log_p_a = log_pdf_normal(a) #log N(z|loc,scale)
    log_p_b = log_pdf_normal(b)  #log N(z|-loc,scale)

    # This formula is dz = N(z|-loc,scale) + N(z|loc,scale)
    log_dz = tf.math.reduce_logsumexp([log_p_b, log_p_a], axis=0)

    # This formula is dloc = N(z|-loc,scale) - N(z|loc,scale)
    log_dloc, dloc_sign = tfm.log_sub_exp(log_p_b, log_p_a)

    # This formula is -a * N(z|-loc, scale) - b * N(z|loc, scale)
    weights = [a, b]
    log_dscale, dscale_sign  = tfm.reduce_weighted_logsum_exp(
            [log_p_b, log_p_a], w=[a, b], axis=0, return_sign=True)

    result = (
        -dloc_sign * tf.exp(log_dloc - log_dz), 
        -dscale_sign * tf.exp(log_dscale - log_z)
    )
    return result

def _cdf_log_space(x, loc, scale):
    x = tf.convert_to_tensor(x)
    ir2 = tf.constant(tf.math.reciprocal(tf.sqrt(2.)), dtype=x.dtype)

    denom = ir2 / scale
    a = denom * (x + loc) 
    b = denom * (x - loc) 

    log_result,sign = tfm.log_sub_exp(
        special_math.log_ndtr(a), 
        special_math.log_ndtr(b),
        return_sign=True,
    )
    return 0.5 * sign * exp(log_result)

def _cdf(x, loc, scale):
    x = tf.convert_to_tensor(x)
    a = (x + loc) / scale
    b = (x - loc) / scale
    ir2 = tf.constant(tf.math.reciprocal(tf.sqrt(2.)), dtype=x.dtype)
    return 0.5 * (tf.math.erf(ir2 * a) - tf.math.erf(ir2 * b))

def cdf(x, loc, scale, log_space=True):
    if log_space:
        return _cdf_log_space(x, loc, scale)
    return _cdf(x, loc, scale)

@tf.custom_gradient
def stateless_folded_normal(shape, loc, scale, seed=None):
  seed = samplers.sanitize_seed(seed)
  z = tf.random.stateless_normal(shape, seed, mean=loc, stddev=scale)
  z = tf.abs(z)
  def grad(upstream):
    dloc, dscale = reparameterization_gradient(z, loc, scale)
    return None, upstream * dloc, upstream * dscale, None
  return z, grad

class FoldedNormal(tfd.TransformedDistribution):
  """The folded normal distribution."""
  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='FoldedNormal'):
    """Construct a folded normal distribution.
    Args:
      loc: Floating-point `Tensor`; the means of the underlying
        Normal distribution(s).
      scale: Floating-point `Tensor`; the stddevs of the underlying
        Normal distribution(s).
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(FoldedNormal, self).__init__(
          distribution=tfd.Normal(loc=loc, scale=scale),
          bijector=abs_bijector.AbsoluteValue(),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        #loc=parameter_properties.ParameterProperties(
        #    default_constraining_bijector_fn=(
        #        lambda: exp_bijector.Exp())),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: exp_bijector.Exp())),
        )
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the pre-transformed mean."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for the pre-transformed standard deviation."""
    return self.distribution.scale

  @property
  def _pi(self):
    return tf.convert_to_tensor(np.pi, self.dtype)

  def _cdf(self, x):
    loc,scale = self.loc,self.scale
    return cdf(x, loc, scale)

  def _sample_n(self, shape=(), seed=None):
    loc = self.loc
    scale = self.scale
    return stateless_folded_normal(shape, loc, scale, seed)

  def _log_prob(self,  value):
    result = super(FoldedNormal, self)._log_prob(value)
    return tf.where(value < 0, tf.constant(-np.inf, dtype=result.dtype), result)

  def _mean(self):
    u = self.loc
    s = self.scale
    snr = u/s
    return s * tf.sqrt(2/self._pi) * tf.math.exp(-0.5 * tf.square(snr)) + u * (1. - 2. * special_math.ndtr(-snr))

  def _variance(self):
    u = self.loc
    s = self.scale
    return tf.square(u) + tf.square(s) - tf.square(self.mean())

  def sample_square(self, sample_shape=(), seed=None, name='sample', **kwargs):
    z = self.distribution.sample(sample_shape, seed, name, **kwargs)
    return tf.square(z)


if __name__=="__main__":
#    n = 100
#    loc,scale = np.random.random((2, n)).astype('float32')
#    loc = tf.Variable(loc)
#    scale = tf.Variable(scale)
#    q = FoldedNormal(loc, scale)
#    p = FoldedNormal(0., 1.)
#
#    x = np.linspace(-10., 10., 1000)
#    u = q.mean()
#    s = q.stddev()
#    v = q.variance()
#
#    q.log_prob(x[:,None])
#    q.prob(x[:,None])
#    q.sample(n)
#    q.kl_divergence(p)
#
    from tensorflow_probability import bijectors as tfb
    from tensorflow_probability import util as tfu
    from tensorflow import keras as tfk

    #loc = tfu.TransformedVariable(1., tfb.Exp())
    loc = tf.Variable(1.)
    scale = tfu.TransformedVariable(1., tfb.Exp())
    q =  FoldedNormal(loc, scale)
    target_q = FoldedNormal(0., 0.1)

    opt = tfk.optimizers.Adam()

    @tf.function
    def loss_fn(q, p, n):
        z = q.sample(n)
        log_q = q.log_prob(z)
        log_p = p.log_prob(z)

        kl_div = log_q - log_p
        loss = tf.reduce_mean(kl_div)
        return loss

    from tqdm import trange
    n = 3
    steps = 10_000
    bar = trange(steps)
    losses = []
    for i in bar:
        with tf.GradientTape() as tape:
            loss = loss_fn(q, target_q, n)
        grads = tape.gradient(loss, q.trainable_variables)
        opt.apply_gradients(zip(grads, q.trainable_variables))
        loss = float(loss)
        losses.append(loss)
        bar.set_description(f"loss: {loss:0.2f}")

    from IPython import embed
    embed(colors='linux')


