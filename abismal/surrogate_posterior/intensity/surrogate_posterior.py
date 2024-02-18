import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.surrogate_posterior import PosteriorBase


def WilsonPrior(centric, epsilon, sigma=1.):
    concentration = tf.where(centric, 0.5, 1.)
    rate = tf.where(centric, 0.5 / sigma / epsilon, 1. / sigma / epsilon)
    return tfd.Gamma(concentration, rate)

class WilsonPosterior(PosteriorBase):
    parameterization = 'intensity'

    def __init__(self, rasu, kl_weight, scale_factor=1e-2, eps=1e-12, concentration_min=1., **kwargs):
        super().__init__(rasu, **kwargs)
        self.prior = WilsonPrior(
            rasu.centric,
            rasu.epsilon,
        )

        self.rasu = rasu

        self.kl_weight = kl_weight
        self.rate = tfu.TransformedVariable(
            5. * tf.ones_like(rasu.centric, dtype='float32'),
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.concentration = tfu.TransformedVariable(
            2. * tf.ones_like(rasu.centric, dtype='float32'),
            tfb.Chain([
                tfb.Shift(concentration_min + eps), 
                #tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

    def flat_prior(self):
        return self.prior

    def flat_distribution(self):
        return tfd.Gamma(self.concentration, self.rate)

    def sqrt_gamma_mean_std(self, q):
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

    def to_datasets(self, seen=True):
        h,k,l = self.rac.Hunique.numpy().T
        q = self.flat_distribution()
        I = q.mean()      
        SIGI = q.stddev()
        F,SIGF = self.sqrt_gamma_mean_std(q)
        asu_id = self.rac.asu_id
        for i,rasu in enumerate(self.rac):
            idx = self.rac.asu_id.numpy() == i
            if seen:
                idx = idx & self.seen.numpy()

            out = rs.DataSet({
                'H' : rs.DataSeries(h, dtype='H'),
                'K' : rs.DataSeries(k, dtype='H'),
                'L' : rs.DataSeries(l, dtype='H'),
                'I' : rs.DataSeries(I, dtype='J'),
                'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
                'F' : rs.DataSeries(F, dtype='F'),
                'SIGF' : rs.DataSeries(SIGF, dtype='Q'),
                },
                merged=True,
                cell=rasu.cell,
                spacegroup=rasu.spacegroup,
            )[idx]

            out = out.set_index(['H', 'K', 'L'])
            if rasu.anomalous:
                out = out.unstack_anomalous()
                out = out[[
                    'I(+)',
                    'SIGI(+)',
                    'I(-)',
                    'SIGI(-)',
                    'F(+)',
                    'SIGF(+)',
                    'F(-)',
                    'SIGF(-)',
                ]]
            yield out


    def register_kl(self, ipred=None, asu_id=None, hkl=None, training=None):
        if training:
            q,p = self.flat_distribution(), self.flat_prior()
            kl_div = q.kl_divergence(p)
            kl_div = tf.reduce_mean(kl_div)
            self.add_metric(kl_div, name='KL')
            self.add_loss(self.kl_weight * kl_div)


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

