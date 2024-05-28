import tensorflow as tf
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import dtype_util
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
import numpy as np
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.bijectors import exp as exp_bijector

class Rice(distribution.AutoCompositeTensorDistribution):
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

    @property
    def nu(self):
        return self._nu

    @property
    def sigma(self):
        return self._sigma

    @property
    def _pi(self):
        return tf.convert_to_tensor(np.pi, self.dtype)

    def sample_square(self, sample_shape=(), seed=None, name='sample', **kwargs):
        base_normal = tfd.Normal(0., self.sigma)
        s1 = base_normal.sample(sample_shape=sample_shape, seed=seed, name=name, **kwargs)
        s2 = base_normal.sample(sample_shape=sample_shape, seed=seed, name=name, **kwargs)
        return tf.square(s1) + tf.square(s2 + self.nu)

    def _sample_n(self, n, seed=None):
        sqr = self.sample_square(n, seed)
        return tf.math.sqrt(sqr)

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

    @property
    def _bivariate_normal(self):
        loc   = tf.convert_to_tensor([1., 0.], dtype=self.dtype)[...,None,:] * self.nu[...,None] 
        scale = tf.convert_to_tensor([1., 1.], dtype=self.dtype)[...,None,:] * self.sigma[...,None] 
        mvn = tfd.MultivariateNormalDiag(loc, scale)
        return mvn

@kullback_leibler.RegisterKL(Rice, Rice)
def _kl_rice_rice(q, p, name=None):
    return q._bivariate_normal.kl_divergence(p._bivariate_normal)

if __name__=="__main__":
    n = 100
    loc,scale = np.random.random((2, n)).astype('float32')
    q = Rice(loc, scale)
    p = Rice(0., 1.)

    x = np.linspace(-10., 10., 1000)
    u = q.mean()
    s = q.stddev()
    v = q.variance()

    q.log_prob(x[:,None])
    q.prob(x[:,None])
    q.sample(n)

    from IPython import embed
    embed(colors='linux')
