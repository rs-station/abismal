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
from abismal.distributions.truncated_normal import TruncatedNormal


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

class TrXXuncatedNormal(tfd.TruncatedNormal):
    def sample(self, *args, **kwargs):
        z = super().sample(*args, **kwargs)
        safe = tf.maximum(z, self.low)
        return safe

class WilsonBase(PosteriorBase):
    parameterization = 'structure_factor'
    prior_scale = 1.

    def flat_prior(self):
        prior = WilsonPrior(
            self.rac.centric,
            self.rac.epsilon,
            self.prior_scale,
        )
        return prior

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
            q_z = q.log_prob(ipred) 
            p_z = p.log_prob(ipred)
            kl_div = q_z - p_z
            kl_div = tf.reduce_mean(kl_div)
            self.add_metric(kl_div, name='KL')
            #self.add_loss(self.kl_weight * kl_div)
            #self.add_metric(tf.reduce_mean(q_z), name='q_z')
            #self.add_metric(tf.reduce_mean(p_z), name='p_z')
        return kl_div




class WilsonPosterior(WilsonBase):
    def __init__(self, rac, scale_factor=1e-1, epsilon=1e-32, kl_weight=1., prior_scale=1., **kwargs):
        super().__init__(rac, epsilon=epsilon, kl_weight=kl_weight, **kwargs)
        self.low = self.epsilon * tf.cast(~self.rac.centric, dtype='float32')
        p = self.flat_prior()
        self.prior_scale = prior_scale
        self.loc = tfu.TransformedVariable(
            p.mean(),
            tfb.Chain([
                tfb.Shift(self.low + epsilon), 
                tfb.Exp(),
            ]),
        )

        self.scale = tfu.TransformedVariable(
            scale_factor * p.stddev(),
            tfb.Chain([
                tfb.Shift(self.low + epsilon), 
                tfb.Exp(),
            ]),
        )

    @property
    def high(self):
        high = self.loc + 100. * self.scale
        return high

    def flat_distribution(self):
        q = TruncatedNormal(
            self.loc, 
            self.scale, 
            self.low,
        )
        return q

class NeuralWilsonPosterior(WilsonBase):
    def __init__(self, rac, mlp, dmodel, epsilon=1e-32, kl_weight=1., kernel_initializer=None, **kwargs):
        super().__init__(rac, epsilon=epsilon, kl_weight=kl_weight, **kwargs)

        self.input_layer  = tfk.layers.Dense(dmodel, kernel_initializer=kernel_initializer)
        self.output_layer = tfk.layers.Dense(2)
        self.mlp = mlp

        self.low = self.epsilon * tf.cast(~self.rac.centric, dtype='float32')
        p = self.flat_prior()

        self.loc_bijector = tfb.Chain([
            tfb.Shift(self.low + epsilon),
            tfb.Exp(),
        ])
        self.scale_bijector = tfb.Chain([
            tfb.Shift(self.low + 1e-4*tf.reduce_mean(p.stddev())),
            tfb.Exp(),
        ])

    def flat_distribution(self):
        loc,scale = self._loc_and_scale()
        high = loc + 100. * scale
        #tf.print("MEAN LOC: {}".format(tf.reduce_mean(loc)))
        #tf.print("MEAN SCALE: {}".format(tf.reduce_mean(scale)))
        q = TruncatedNormal(
            loc, 
            scale, 
            self.low,
        )
        return q

    def _loc_and_scale(self):
        self.rac.Hunique.numpy()
        self.rac.asu_id

        aid = tf.cast(self.rac.asu_id[...,None], 'float32')
        hkl = tf.cast(self.rac.Hunique, 'float32')
        AHKL = tf.concat((aid, hkl), axis=-1)
        a = tf.math.reduce_min(AHKL, axis=-2, keepdims=True)
        b = tf.math.reduce_max(AHKL, axis=-2, keepdims=True)
        denom = b - a
        denom = tf.where(denom == 0., 1., denom)
        p = 2. * (AHKL - a) / denom  - 1.

        num_frequencies = 5
        d = p.get_shape()[-1]
        p = tf.repeat(p, num_frequencies, axis=-1)

        L = tf.range(0, num_frequencies, dtype='float32')
        f = np.pi * 2 ** L
        f = tf.tile(f, (d,))
        fp = f * p

        encoded = tf.concat((
            tf.sin(fp),
            tf.cos(fp),
        ), axis=-1)

        loc,scale = tf.unstack(self.output_layer(self.mlp(self.input_layer(encoded))), axis=-1)
        loc = self.loc_bijector(loc)
        scale = self.scale_bijector(scale)
        return loc,scale

    @property
    def loc(self):
        loc,_ = self._loc_and_scale()
        return loc

    @property
    def scale(self):
        _,scale = self._loc_and_scale()
        return scale


