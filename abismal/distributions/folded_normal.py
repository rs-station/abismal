import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import prefer_static as ps
import numpy as np
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers
from tensorflow_probability import math as tfm
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability import math as tfm
import math

def folded_normal_sample_gradients(z, loc, scale):
    alpha = (z + loc) / scale
    beta = (z - loc) / scale

    log_p_a = tfd.Normal(loc, scale).log_prob(z)   #N(z|loc,scale)
    log_p_b = tfd.Normal(-loc, scale).log_prob(z)  #N(z|-loc,scale)

    p_a = tf.math.exp(log_p_a)
    p_b = tf.math.exp(log_p_b)

    # This formula is dz = N(z|-loc,scale) + N(z|loc,scale)
    log_dz = tfm.log_add_exp(log_p_a, log_p_b)
    dz = tf.math.exp(log_dz)

    # This formula is dloc = N(z|-loc,scale) - N(z|loc,scale)
    log_dloc,sign  = tfm.log_sub_exp(log_p_b, log_p_a, return_sign=True)
    #dloc = p_b - p_a
    log_dloc = log_dloc - log_dz
    dloc = -sign * tf.math.exp(log_dloc)
    #dloc = -dloc / dz

    # This formula is -[b * N(z|-loc, scale) + a * N(z|loc, scale)]
    dscale = -alpha * p_b - beta * p_a
    dscale = -dscale / dz
    return dloc, dscale

@tf.custom_gradient
def stateless_folded_normal(shape, loc, scale, seed):
    z = tf.random.stateless_normal(shape, seed, mean=loc, stddev=scale)
    z = tf.abs(z)
    def grad(upstream):
        dloc,dscale = folded_normal_sample_gradients(z, loc, scale)
        dloc = tf.reduce_sum(upstream * dloc, axis=0)
        dscale = tf.reduce_sum(upstream * dscale, axis=0)
        return None, dloc, dscale, None
    return z, grad

class FoldedNormal(tfd.Distribution):
    """The folded normal distribution."""
    _normal_crossover = 10.
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

    def _batch_shape_tensor(self, loc=None, scale=None):
        if loc is None:
            loc = self.loc
        if scale is None:
            scale =self.scale
        return array_ops.shape(loc / scale)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            loc=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                )
            ),
        )

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the pre-transformed standard deviation."""
        return self._scale

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _cdf(self, x):
        # F(x) = Phi((x - loc)/scale) + Phi((x + loc)/scale) - 1, which in terms of
        # erf is the *sum* below. This read `erf(a) - erf(b)` until 2026-08-24,
        # which is a different function entirely -- at loc=scale=1, x=0.5 it gave
        # 0.6247 where the folded normal CDF is 0.2417. Nothing in abismal calls
        # cdf(), so no result depended on it, but it agreed with neither _log_prob
        # nor the sampler's implicit-reparameterization gradients, both of which
        # are and were correct.
        loc, scale = self.loc, self.scale
        x = tf.convert_to_tensor(x)
        a = (x + loc) / scale
        b = (x - loc) / scale
        # Build the constant at x's dtype rather than computing it in float32 and
        # casting, which cost ~2e-8 relative error in a float64 distribution.
        ir2 = tf.constant(1.0 / math.sqrt(2.0), dtype=x.dtype)
        cdf = 0.5 * (tf.math.erf(ir2 * a) + tf.math.erf(ir2 * b))
        # The support is non-negative; without this the expression runs negative
        # below zero (-0.477 at x=-1, loc=scale=1) rather than returning 0.
        return tf.where(x < 0, tf.zeros_like(cdf), cdf)

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

    @staticmethod
    def _folded_normal_mean(loc, scale):
        u,s = loc, scale
        c = loc / scale
        mean = (
            s * tf.sqrt(2/math.pi) * tf.math.exp(-0.5 * c * c) + 
            u * tf.math.erf(c/math.sqrt(2))
        )
        return mean

    def _mean(self):
        u = self.loc
        s = self.scale
        c = u/s

        idx = tf.abs(c) >= self._normal_crossover
        s_safe = tf.where(idx, 1., s)
        u_safe = tf.where(idx, 1., u)

        return tf.where(
            idx,
            tf.abs(u),
            self._folded_normal_mean(u_safe, s_safe)
        )

    def _variance(self):
        u = self.loc
        s = self.scale
        c = u/s

        idx = tf.abs(c) >= 10.
        s_safe = tf.where(idx, 1., s)
        u_safe = tf.where(idx, 1., u)
        m = self._folded_normal_mean(u_safe, s_safe)

        return tf.where(
            idx, 
            s*s,
            u*u + s*s - m*m,
        )

    def moment(self, t):
        """ Use Scipy to calculate the t-moment of a folded normal """
        from scipy.stats import foldnorm,norm
        loc,scale = self.loc.numpy(),self.scale.numpy()
        c = loc / scale
        idx = np.abs(c) > self._normal_crossover
        c_safe = tf.where(idx, c, 0.)
        scale_safe = tf.where(idx, scale, 1.)

        result = np.where(
            idx,
            foldnorm.moment(t, loc/scale, scale=scale),
            norm.moment(t, loc=loc, scale=scale),
        )
        return result

class ExplicitFoldedNormal(FoldedNormal):
    def _sample_n(self, n, seed=None):
        seed = samplers.sanitize_seed(seed)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], axis=0)
        z = tf.random.stateless_normal(shape, seed, mean=loc, stddev=scale)
        z = tf.abs(z)
        return z

