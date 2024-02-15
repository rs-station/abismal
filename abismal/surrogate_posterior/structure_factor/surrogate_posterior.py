import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.distributions import RiceWoolfson
from abismal.surrogate_posterior import PosteriorBase


class Centric(tfd.HalfNormal):
    def __init__(self, epsilon, sigma=1.):
        self.epsilon = tf.convert_to_tensor(epsilon)
        self.sigma = tf.convert_to_tensor(sigma)
        super().__init__(tf.math.sqrt(epsilon * self.sigma))


class Acentric(tfd.Weibull):
    def __init__(self, epsilon, sigma=1.):
        self.epsilon = tf.convert_to_tensor(epsilon)
        self.sigma = tf.convert_to_tensor(sigma)
        super().__init__(
            2., 
            tf.math.sqrt(self.epsilon * self.sigma),
        )


class WilsonPrior(object):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, centric, epsilon, sigma=1.):
        """
        Parameters
        ----------
        centric : array
            Floating point or boolean array with value 1/True for centric reflections and 0/False. for acentric.
        epsilon : array
            Floating point array with multiplicity values for each structure factor.
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. 
        """
        self.epsilon = np.array(epsilon, dtype=np.float32)
        self.centric = np.array(centric, dtype=bool)
        self.sigma = np.array(sigma, dtype=np.float32)

        self.p_centric = Centric(self.epsilon, self.sigma)
        self.p_acentric = Acentric(self.epsilon, self.sigma)

    def log_prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        return tf.where(self.centric, self.p_centric.log_prob(x), self.p_acentric.log_prob(x))

    def prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        return tf.where(self.centric, self.p_centric.prob(x), self.p_acentric.prob(x))

    def mean(self):
        return tf.where(self.centric, self.p_centric.mean(), self.p_acentric.mean())

    def stddev(self):
        return tf.where(self.centric, self.p_centric.stddev(), self.p_acentric.stddev())

    def sample(self, *args, **kwargs):
        #### BLEERRRGG #####
        return tf.where(
            self.centric, 
            self.p_centric.sample(*args, **kwargs),
            self.p_acentric.sample(*args, **kwargs),
        )

class TruncatedNormal(tfd.TruncatedNormal):
    def sample(self, *args, **kwargs):
        z = super().sample(*args, **kwargs)
        safe = tf.maximum(z, self.low)
        return safe

class WilsonPosterior(PosteriorBase):
    parameterization = 'structure_factor'
    high = 1e10

    def __init__(self, rac, scale_factor=1e-1, epsilon=1e-32, kl_weight=1., **kwargs):
        super().__init__(rac, epsilon=epsilon, kl_weight=kl_weight, **kwargs)
        self.low = self.epsilon * tf.cast(~self.rac.centric, dtype='float32')
        p = self.flat_prior()
        self.loc = tfu.TransformedVariable(
            p.mean(),
            tfb.Chain([
                tfb.Shift(self.low + epsilon), 
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.scale = tfu.TransformedVariable(
            scale_factor * p.stddev(),
            tfb.Chain([
                tfb.Shift(self.low + epsilon), 
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
        q = tfd.TruncatedNormal(
            self.loc, 
            self.scale, 
            self.low,
            self.high
        )
        return q

    def to_datasets(self, seen=True):
        h,k,l = self.rac.Hunique.numpy().T
        q = self.flat_distribution()
        F = q.mean()      
        SIGF = q.stddev()
        #This is exact
        I = SIGF*SIGF + F*F
        #This is an approximation based on uncertainty propagation
        SIGI = np.abs(2*F*SIGF)
        asu_id = self.rac.asu_id
        for i,rasu in enumerate(self.rac):
            idx = self.rac.asu_id.numpy() == i
            if seen:
                idx = idx & self.seen.numpy()

            out = rs.DataSet({
                'H' : rs.DataSeries(h, dtype='H'),
                'K' : rs.DataSeries(k, dtype='H'),
                'L' : rs.DataSeries(l, dtype='H'),
                'F' : rs.DataSeries(F, dtype='F'),
                'SIGF' : rs.DataSeries(SIGF, dtype='Q'),
                'I' : rs.DataSeries(I, dtype='J'),
                'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
                },
                merged=True,
                cell=rasu.cell,
                spacegroup=rasu.spacegroup,
            )[idx]

            out = out.set_index(['H', 'K', 'L'])
            if rasu.anomalous:
                out = out.unstack_anomalous()
                out = out[[
                    'F(+)',
                    'SIGF(+)',
                    'F(-)',
                    'SIGF(-)',
                    'I(+)',
                    'SIGI(+)',
                    'I(-)',
                    'SIGI(-)',
                ]]
            yield out

    def register_kl(self, ipred=None, asu_id=None, hkl=None, training=None):
        kl_div = 0.
        if training:
            p = self.flat_prior()
            q = self.flat_distribution()
            kl_div = q.log_prob(ipred) - p.log_prob(ipred)
            kl_div = tf.reduce_mean(kl_div)
            self.add_metric(kl_div, name='KL')
            self.add_loss(self.kl_weight * kl_div)
        return kl_div

