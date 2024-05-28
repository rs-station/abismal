from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.intensity.wilson import WilsonPrior
import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk


class GammaPosterior(IntensityPosteriorBase):
    def __init__(self, rac, kl_weight, scale_factor=1e-2, eps=1e-12, concentration_min=1., **kwargs):
        super().__init__(rac, **kwargs)
        self.rac = rac

        self.kl_weight = kl_weight
        self.rate = tfu.TransformedVariable(
            5. * tf.ones_like(rac.centric, dtype='float32'),
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.concentration = tfu.TransformedVariable(
            2. * tf.ones_like(rac.centric, dtype='float32'),
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

    def flat_distribution(self):
        return tfd.Gamma(self.concentration, self.rate)

    def get_flat_fsigf(self, q):
        """
        Compute the mean and standard deviation of the square root of a gamma distribution.
        """
        alpha = q.concentration
        beta  = q.rate
        m = alpha
        omega = alpha / beta
        mean = tf.math.exp(
            tf.math.lgamma(m + 0.5) - tf.math.lgamma(m) + 0.5 * tf.math.log(omega) - 0.5 * tf.math.log(m)
        )
        var = omega - mean*mean
        std = tf.math.sqrt(var)
        return mean,std


if __name__=="__main__":
    def sqrt_moments(gamma):
        k = alpha = gamma.concentration
        beta  = gamma.rate
        theta = tf.reciprocal(beta)

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
        log_std = 2

        return tf.exp(log_mean), tf.exp(log_std)

