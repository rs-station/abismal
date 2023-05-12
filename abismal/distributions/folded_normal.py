import tensorflow as tf
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
import numpy as np
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import absolute_value as abs_bijector
from tensorflow_probability.python.internal import parameter_properties


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
        loc=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: exp_bijector.Exp())),
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
    n = 100
    loc,scale = np.random.random((2, n)).astype('float32')
    q = FoldedNormal(loc, scale)
    p = FoldedNormal(0., 1.)

    x = np.linspace(-10., 10., 1000)
    u = q.mean()
    s = q.stddev()
    v = q.variance()

    q.log_prob(x[:,None])
    q.prob(x[:,None])
    q.sample(n)
    q.kl_divergence(p)

    from IPython import embed
    embed(colors='linux')

