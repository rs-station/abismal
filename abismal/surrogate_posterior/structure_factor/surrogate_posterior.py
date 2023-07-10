import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.distributions import RiceWoolfson,FoldedNormal
from abismal.surrogate_posterior import WilsonBase,PosteriorCollectionBase


def WilsonPrior(centric, multiplicity, sigma=1.):
    scale = tf.where(centric, tf.sqrt(multiplicity), tf.sqrt(0.5 * multiplicity))
    loc = tf.zeros_like(multiplicity)
    return RiceWoolfson(loc, scale, centric)

class PosteriorCollection(PosteriorCollectionBase):
    parameterization = 'structure_factor'

    def call(self, asu_id, hkl, training=None, mc_samples=1):
        asu_id = tf.squeeze(asu_id, axis=-1)
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
        multiplicity = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.rasu.gather(wp.rasu.epsilon, h),
        )
        q = RiceWoolfson(loc, scale, centric)
        p = WilsonPrior(centric, multiplicity)

        if training:
            for i,wp in enumerate(self.posteriors):
                mask = asu_id == i
                _hkl = hkl[mask] 
                if tf.math.reduce_any(mask):
                    wp._register_seen(_hkl)

        kl_weight = self.posteriors[0].kl_weight
        kl_div = q.kl_divergence(p)
        kl_div = tf.reduce_mean(kl_div)
        self.add_metric(kl_div, name='KL')
        self.add_loss(kl_weight * kl_div)

        return q

class WilsonPosterior(WilsonBase):
    parameterization = 'structure_factor'
    def __init__(self, rasu, kl_weight, scale_factor=1e-1, eps=1e-12, **kwargs):
        super().__init__(rasu, kl_weight=kl_weight, **kwargs)
        self.prior = WilsonPrior(
            rasu.centric,
            rasu.epsilon,
        )

        self.rasu = rasu

        self.loc = tfu.TransformedVariable(
            tf.ones_like(rasu.centric, dtype='float32'),
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
        #This is exact
        I = SIGF*SIGF + F*F
        #This is an approximation based on uncertainty propagation
        SIGI = np.abs(2*F*SIGF)
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
            cell=self.rasu.cell,
            spacegroup=self.rasu.spacegroup,
        )
        return out

    def to_dataset(self, seen=True):
        """
        Parameters
        ----------
        seen : bool (optional)
            Only include reflections seen during training. Defaults to True. 
        """
        out = self._to_dataset()
        if seen:
            out = out[self.seen.numpy()]
        out = out.set_index(['H', 'K', 'L'])

        if self.rasu.anomalous:
            out = out.unstack_anomalous()
            out = out[[
                'F(+)', 'SIGF(+)', 'F(-)', 'SIGF(-)', 'I(+)', 'SIGI(+)', 'I(-)', 'SIGI(-)'
                ]]
        return out

    def call(self, hkl, mc_samples=1, training=None):
        q,p = self.flat_distribution, self.prior
        kl_div = q.kl_divergence(p)
        kl_div = tf.reduce_mean(kl_div)
        self.add_metric(kl_div, name='KL_{}')
        self.add_loss(self.kl_weight * kl_div)
        if training:
            self._register_seen(hkl)

        q = self.distribution(hkl)
        return q

