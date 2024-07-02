from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.intensity.wilson import WilsonPrior
import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk


class GammaPosterior(IntensityPosteriorBase):
    def __init__(self, rac, kl_weight, scale_factor=1e-2, eps=1e-12, concentration_min=1., **kwargs):
        super().__init__(rac, **kwargs)
        self.rac = rac

        self.kl_weight = kl_weight
        loc = self.flat_prior().mean()
        scale = scale_factor * loc

        rate_init = tf.square(loc / scale)
        conc_init = loc / tf.square(scale)

        self.rate = tfu.TransformedVariable(
            rate_init,
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.concentration = tfu.TransformedVariable(
            conc_init,
            tfb.Chain([
                tfb.Shift(concentration_min + eps), 
                #tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

    def flat_prior(self):
        prior = WilsonPrior(
            self.rac.centric,
            self.rac.epsilon,
        )
        return prior

    def flat_distribution(self):
        return tfd.Gamma(self.concentration, self.rate)

    def get_flat_fsigf(q, eps=1e-6):
        """
        Compute the mean and standard deviation of the square root of a gamma distribution.
        """
        alpha = q.concentration
        beta  = q.rate
        omega = alpha / beta

        log_mean_sqrt_beta = tf.math.lgamma(alpha + 0.5) - tf.math.lgamma(alpha) #log(mean * beta**0.5)
        mean = tf.math.exp(log_mean_sqrt_beta) / tf.sqrt(beta)

        num = alpha - tf.math.exp(2.* log_mean_sqrt_beta)
        var = tf.where(
            alpha > 1e4,
            0.25/beta, #The limit is 0.25 for num
            num/beta,
        )

        std = tf.math.sqrt(var)
        return mean,std



if __name__=="__main__":
    def sqrt_moments(gamma):
        k = alpha = gamma.concentration
        beta  = gamma.rate
        theta = tf.math.reciprocal(beta)

        """
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
