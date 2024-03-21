import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import prefer_static as ps
import numpy as np
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import absolute_value as abs_bijector
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers
from tensorflow_probability import math as tfm
from tensorflow_probability.python.internal import tensor_util

def sample_gradients(z, loc, scale):
    alpha = (z + loc) / scale
    beta = (z - loc) / scale

    p_a = tfd.Normal(loc, scale).prob(z)   #N(z|loc,scale)
    p_b = tfd.Normal(-loc, scale).prob(z)  #N(z|-loc,scale)

    # This formula is dz = N(z|-loc,scale) + N(z|loc,scale)
    dz = p_a + p_b

    # This formula is dloc = N(z|-loc,scale) - N(z|loc,scale)
    dloc = p_b - p_a

    # This formula is -[b * N(z|-loc, scale) + a * N(z|loc, scale)]
    dscale = -alpha * p_b - beta * p_a
    return dz, dloc, dscale


@tf.custom_gradient
def stateless_folded_normal(shape, loc, scale, seed):
    z = tf.random.stateless_normal(shape, seed, mean=loc, stddev=scale)
    z = tf.abs(z)
    def grad(upstream):
        grads = sample_gradients(z, loc, scale)
        dz,dloc,dscale = grads[0], grads[1], grads[2]
        dloc = tf.reduce_sum(-upstream * dloc / dz, axis=0)
        dscale = tf.reduce_sum(-upstream * dscale / dz, axis=0)
        return None, dloc, dscale, None
    return z, grad

class FoldedNormal(tfd.Distribution):
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
          dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
          self._loc = tensor_util.convert_nonref_to_tensor(
              loc, dtype=dtype, name='loc')
          self._scale = tensor_util.convert_nonref_to_tensor(
              scale, dtype=dtype, name='scale')
          super(FoldedNormal, self).__init__(
              dtype=dtype,
              reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

    def _batch_shape_tensor(self, loc, scale):
        return array_ops.shape(loc / scale)

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
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the pre-transformed standard deviation."""
        return self._scale

    @property
    def _pi(self):
        return tf.convert_to_tensor(np.pi, self.dtype)

    def _cdf(self, x):
        loc,scale = self.loc,self.scale
        x = tf.convert_to_tensor(x)
        a = (x + loc) / scale
        b = (x - loc) / scale
        ir2 = tf.constant(tf.math.reciprocal(tf.sqrt(2.)), dtype=x.dtype)
        return 0.5 * (tf.math.erf(ir2 * a) - tf.math.erf(ir2 * b))

    def _sample_n(self, n, seed=None):
        seed = samplers.sanitize_seed(seed)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], axis=0)
        return stateless_folded_normal(shape, loc, scale, seed)

    def _log_prob(self,  value):
        loc,scale = self.loc,self.scale
        result = tfm.log_add_exp(
            tfd.Normal(loc, scale).log_prob(value), 
            tfd.Normal(-loc, scale).log_prob(value),
        )
        return result
        #return tf.where(value < 0, tf.constant(-np.inf, dtype=result.dtype), result)

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
    d = ()
    #loc = tfu.TransformedVariable(tf.ones(d), tfb.Exp())
    loc = tf.Variable(tf.ones(d))
    scale = tfu.TransformedVariable(
        tf.ones(d), 
        tfb.Chain([
            tfb.Shift(1e-6),
            tfb.Exp(),
        ]),
    )
    q =  FoldedNormal(loc, scale)
    target_q = FoldedNormal(0., 0.1)
    q.sample(32)

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
    n = 128
    steps = 2_000
    bar = trange(steps)
    losses = []
    locs = []
    scales = []
    for i in bar:
        with tf.GradientTape() as tape:
            loss = loss_fn(q, target_q, n)
        grads = tape.gradient(loss, q.trainable_variables)
        opt.apply_gradients(zip(grads, q.trainable_variables))
        loss = float(loss)
        locs.append(float(q.loc))
        scales.append(float(tf.convert_to_tensor(q.scale)))
        _scale = float(tf.convert_to_tensor(q.scale))
        _loc = float(tf.convert_to_tensor(q.loc))
        locs.append(_loc)
        scales.append(_scale)
        losses.append(loss)
        bar.set_description(f"loss: {loss:0.2f} loc: {float(_loc):0.2f} scale: {_scale:0.2f}")

    from IPython import embed
    embed(colors='linux')


