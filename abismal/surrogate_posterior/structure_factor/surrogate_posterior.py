import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.distributions import RiceWoolfson
from abismal.surrogate_posterior import WilsonBase,PosteriorCollectionBase


class PosteriorCollection(PosteriorCollectionBase):
    def call(self, asu_id, hkl, training=None):
        loc = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.rasu.gather(wp.loc, h),
        )
        scale = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.rasu.gather(wp.scale, h),
        )
        centric = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.rasu.gather(wp.rasu.centric, h),
        )
        for i,wp in enumerate(self.posteriors):
            _hkl = hkl[tf.squeeze(asu_id, axis=-1) == i] #srsly? y squeeze?
            if training:
                wp._register_seen(_hkl)
                wp.kl_div(_hkl, training=training)

        q = RiceWoolfson(loc, scale, centric)
        return q

def WilsonPrior(centric, multiplicity, sigma=1.):
    scale = tf.where(centric, tf.sqrt(multiplicity), tf.sqrt(0.5 * multiplicity))
    loc = tf.zeros_like(multiplicity)
    return RiceWoolfson(loc, scale, centric)

class WilsonPosterior(WilsonBase):
    def __init__(self, rasu, kl_weight, scale_factor=1e-2, eps=1e-12, **kwargs):
        super().__init__(rasu, **kwargs)
        self.prior = WilsonPrior(
            rasu.centric,
            rasu.epsilon,
        )

        self.rasu = rasu

        self.kl_weight = kl_weight
        self.loc = tfu.TransformedVariable(
            5. * tf.ones_like(rasu.centric, dtype='float32'),
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.scale = tfu.TransformedVariable(
            scale_factor * tf.ones_like(rasu.centric, dtype='float32'),
            tfb.Chain([
                #tfb.Shift(1. + eps), 
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

    @property
    def flat_distribution(self):
        return RiceWoolfson(self.loc, self.scale, self.rasu.centric)

    def distribution(self, hkl):
        loc = self.rasu.gather(self.loc, hkl)
        scale = self.rasu.gather(self.scale, hkl)
        centric = self.rasu.gather(self.rasu.centric, hkl)
        return RiceWoolfson(loc, scale, self.rasu.centric)

    def _to_dataset(self):
        h,k,l = self.rasu.Hunique.T
        q = self.flat_distribution
        F = q.mean()      
        SIGF = q.stddev()
        #This is an approximation based on uncertainty propagation
        I = SIGF*SIGF + F*F
        SIGI = np.abs(2*F*SIGF)
        out = rs.DataSet({
            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
            'F' : rs.DataSeries(I, dtype='J'),
            'SIGF' : rs.DataSeries(SIGI, dtype='Q'),
            'I' : rs.DataSeries(I, dtype='J'),
            'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
            },
            merged=True,
            cell=self.rasu.cell,
            spacegroup=self.rasu.spacegroup,
        )
        return out

    def call(self, hkl, mc_samples=1, training=None):
        if training:
            q,p = self.flat_distribution, self.prior
            kl_div = q.kl_divergence(p)
            kl_div = tf.reduce_mean(kl_div)
            self.add_metric(kl_div, name='KL')
            self.add_loss(self.kl_weight * kl_div)
            self._register_seen(hkl)

        q = self.distribution(hkl)
        return q

