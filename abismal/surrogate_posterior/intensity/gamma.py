from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.gamma import GammaPosteriorBase
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk


@tfk.saving.register_keras_serializable(package="abismal")
class GammaPosterior(GammaPosteriorBase, IntensityPosteriorBase):
    """
    an intensity surrogate posterior parameterized by a gamma distribution.
    """
    def get_flat_fsigf(self):
        """
        the square root of a gamma distribution is the nakagami distribution. 
        compute the mean and standard deviation of the square root of a gamma distribution.
        see [https://en.wikipedia.org/wiki/nakagami_distribution#random_variate_generation]. 
        """
        q = self.flat_distribution()
        conc = q.concentration
        rate  = q.rate

        # log_gamma(conc + 0.5) - log_gamma(conc)
        ldiff = -tfm.log_gamma_difference(0.5, conc)

        mean = tf.math.exp(
            ldiff - 0.5*tf.math.log(rate)
        )
        var = (conc - tf.math.exp(2.* ldiff)) / rate
        std = tf.math.sqrt(var)
        var = tf.where(var <= 0., self.epsilon * self.epsilon, var)
        std = tf.math.sqrt(var)
        return mean,std


if __name__=="__main__":
    def sqrt_moments(gamma):
        k = alpha = gamma.concentration
        beta  = gamma.rate
        theta = tf.math.reciprocal(beta)

        r"""
        Random wikipedia knowledge:
        X ~ Gamma(k, θ), then \sqrt{X} follows a generalized gamma distribution with parameters 
        p = 2, d = 2k, and a = \sqrt{\theta}
        """
        p = 2.
        d = 2. * k
        a = tf.sqrt(theta)
        la = tf.math.log(a)
        log_mean = la + tf.math.lgamma((d + 1) / p) - tf.math.lgamma(d / p)
        log_t = 2.*la + tf.math.lgamma((d + 2) / p) - tf.math.lgamma(d / p)
        var = tf.exp(log_t) - tf.exp(2.*log_mean)
        std = tf.sqrt(var)
        mean = tf.exp(log_mean)
        return tf.exp(log_mean), tf.sqrt(var)


    n = 1000
    loc = np.random.random(n)
    scale = np.random.random(n)
    #scale = 0.0001 * loc

    rate = tf.square(loc / scale)
    conc = loc / tf.square(scale)

    q = tfd.Gamma(conc, rate)
    m,s = get_flat_fsigf(q)
    m,s = sqrt_moments(q)
