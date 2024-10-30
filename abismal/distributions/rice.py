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
from tensorflow_probability import math as tfm
import math

def rice_sample_gradients(z, nu, sigma):
    nuzsig2 = nu*z / (sigma*sigma)
    dzdnu = tf.math.bessel_i1e(nuzsig2) / tf.math.bessel_i0e(nuzsig2)
    dzdsigma = (z - nu * dzdnu)/sigma
    return dzdnu, dzdsigma

@tf.custom_gradient
def stateless_rice(shape, nu, sigma, seed):
    z1 = tf.random.stateless_normal(shape, seed, mean=nu, stddev=sigma)
    z2 = tf.random.stateless_normal(shape, seed, mean=0., stddev=sigma)
    z = tf.sqrt(z1*z1 + z2*z2)
    def grad(upstream):
        dnu,dsigma = rice_sample_gradients(z, nu, sigma)
        dnu = tf.reduce_sum(upstream * dnu, axis=0)
        dsigma = tf.reduce_sum(upstream * dsigma, axis=0)
        return None, dnu, dsigma, None
    return z, grad

class Rice(tfd.Distribution):
    """The Rice distribution."""
    def __init__(self,
		   nu,
		   sigma,
		   validate_args=False,
		   allow_nan_stats=True,
		   name='Rice'):

        parameters = dict(locals())
        #Value of nu/sigma for which, above with the pdf and moments will be swapped with a normal distribution
        self._normal_crossover = 40. 
        dtype = dtype_util.common_dtype([nu, sigma], dtype_hint=tf.float32)
        with tf.name_scope(name) as name:
            self._nu = tensor_util.convert_nonref_to_tensor(nu)
            self._sigma = tensor_util.convert_nonref_to_tensor(sigma)
            super(Rice, self).__init__(
                dtype=dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                parameters=parameters,
                name=name)

    def _batch_shape_tensor(self, nu=None, sigma=None):
        if nu is None:
            nu = self.nu
        if sigma is None:
            sigma =self.sigma
        return array_ops.shape(nu / sigma)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
      # pylint: disable=g-long-lambda
        default_bij = lambda: exp_bijector.Exp()
        pp = dict(
            nu=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=default_bij
            ),
            sigma=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=default_bij
            )
        )
        return pp
      # pylint: enable=g-long-lambda

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    @property
    def nu(self):
        return self._nu

    @property
    def sigma(self):
        return self._sigma

    @property
    def _pi(self):
        return tf.convert_to_tensor(np.pi, self.dtype)

    def _sample_n(self, n, seed=None):
        seed = samplers.sanitize_seed(seed)
        nu = tf.convert_to_tensor(self.nu)
        sigma = tf.convert_to_tensor(self.sigma)
        shape = ps.concat([[n], self._batch_shape_tensor(nu=nu, sigma=sigma)], axis=0)
        return stateless_rice(shape, nu, sigma, seed)

    def _log_bessel_i0(self, x):
        return tf.math.log(tf.math.bessel_i0e(x)) + tf.math.abs(x)

    def _log_bessel_i1(self, x):
        return tf.math.log(tf.math.bessel_i1e(x)) + tf.math.abs(x)

    def _bessel_i0(self, x):
        return tf.math.exp(self._log_bessel_i0(x))

    def _bessel_i1(self, x):
        return tf.math.exp(self._log_bessel_i1(x))

    def _laguerre_half(self, x):
        return (1. - x) * tf.math.exp(x / 2. + self._log_bessel_i0(-0.5 * x)) - x * tf.math.exp(x / 2.  + self._log_bessel_i1(-0.5 * x) )

    def _prob(self, X):
        return tf.math.exp(self.log_prob(X))

    def _log_prob(self, X):
        sigma = self.sigma
        nu = self.nu
        log_p = tf.math.log(X) - 2.*tf.math.log(sigma) - (tf.square(X) + tf.square(nu))/(2*tf.square(sigma)) + \
                    self._log_bessel_i0(X * nu/tf.square(sigma))
        return tf.where(X <= 0, -np.inf, log_p)

    def _mean(self):
        sigma = self.sigma
        nu = self.nu
        snr = nu / sigma
        mean = sigma * tf.math.sqrt(self._pi / 2.) * self._laguerre_half(-0.5*tf.square(snr))
        return tf.where(snr > self._normal_crossover,  nu, mean)

    def _variance(self):
        sigma = self.sigma
        nu = self.nu
        snr = nu / sigma
        variance = 2*tf.square(sigma) + tf.square(nu) - 0.5*self._pi * tf.square(sigma) * tf.square(self._laguerre_half(-0.5*tf.square(snr)))
        return tf.where(snr > self._normal_crossover,  tf.square(sigma), variance)

    def _stddev(self):
        return tf.math.sqrt(self.variance())


