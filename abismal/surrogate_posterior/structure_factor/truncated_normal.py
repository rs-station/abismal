import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.distributions import RiceWoolfson,FoldedNormal
from abismal.surrogate_posterior import PosteriorBase


class TruncatedNormal(tfd.TruncatedNormal):
    def sample_intensities(self, *args, **kwargs):
        return tf.square(self.sample(*args, **kwargs))

def WilsonPrior(centric, multiplicity, sigma=1.):
    scale = tf.where(centric, tf.sqrt(multiplicity), tf.sqrt(0.5 * multiplicity))
    loc = tf.zeros_like(multiplicity)
    return RiceWoolfson(loc, scale, centric)

class TruncatedNormalPosteriorCollection(PosteriorCollectionBase):
    parameterization = 'structure_factor'

    def call(self, asu_id, hkl, training=None, mc_samples=1, epsilon=1e-32):
        asu_id = tf.squeeze(asu_id, axis=-1)

        refl_ids = -1
        centric = None
        multiplicity = None
        loc = None
        scale = None
        index_offset = 0
        for i,wp in enumerate(self.posteriors):
            mask = asu_id == i

            if loc is None:
                loc = wp.loc
                scale = wp.scale
                centric = wp.rasu.centric
                multiplicity = wp.rasu.epsilon
            else:
                loc = tf.concat((loc, wp.loc), axis=-1)
                scale = tf.concat((scale, wp.scale), axis=-1)
                centric = tf.concat((centric, wp.rasu.centric), axis=-1)
                multiplicity = tf.concat((multiplicity, wp.rasu.epsilon), axis=-1)

            refl_ids = tf.where(mask, wp.rasu._miller_ids(hkl) + index_offset, refl_ids)
            _hkl = tf.boolean_mask(hkl, mask)

            if tf.math.reduce_any(mask):
                if training:
                    wp._register_seen(_hkl)
            index_offset = index_offset + wp.rasu.asu_size

        #bads = tf.reduce_sum(tf.where(refl_ids == -1, 1., 0.))
        #self.add_metric(bads, name='Missing')
        high = wp.high
        low = tf.where(centric, 0., wp.low)
        q = TruncatedNormal(loc, scale, low, high)
        p = WilsonPrior(centric, multiplicity, 1.)
        z = q.sample(mc_samples)
        kl_div = q.log_prob(z) - p.log_prob(z)
        kl_div = tf.reduce_mean(kl_div)
        kl_weight = wp.kl_weight
        self.add_metric(kl_div, name='KL')
        self.add_loss(kl_weight * kl_div)
        fpred = tf.gather(z, refl_ids, axis=-1)
        return tf.square(fpred)

#        from IPython import embed
#        embed(colors='linux')
#        centric = tf.concat([wp.rasu.centric for wp in self.posteriors], axis=-1)
#        multiplicity = tf.concat([wp.rasu.epsilon for wp in self.posteriors], axis=-1)
#        multiplicity = [wp.rasu.epsilon for wp in self.posteriors]
#        centric = [wp.rasu.centric for wp in self.posteriors]
#
#        centric = self._wp_method_helper(
#            asu_id,
#            hkl,
#            lambda h,wp: wp.rasu.gather(wp.rasu.centric, h),
#        )
#        multiplicity = self._wp_method_helper(
#            asu_id,
#            hkl,
#            lambda h,wp: wp.rasu.gather(wp.rasu.epsilon, h),
#        )
#        loc = self._wp_method_helper(
#            asu_id,
#            hkl,
#            lambda h,wp: wp.rasu.gather(wp.loc, h),
#        )
#        scale = self._wp_method_helper(
#            asu_id,
#            hkl,
#            lambda h,wp: wp.rasu.gather(wp.scale, h),
#        )
#
#        high = wp.high
#        low = tf.where(centric, 0., wp.low)
#        q = TruncatedNormal(loc, scale, low, high)
#        p = WilsonPrior(centric, multiplicity, 1.)
#
#        z = q.sample(mc_samples)
#        kl_div = q.log_prob(z) - p.log_prob(z)
#        kl_div = tf.reduce_mean(kl_div)
#        kl_weight = wp.kl_weight
#        self.add_metric(kl_div, name='KL')
#        self.add_loss(kl_weight * kl_div)
#        return tf.square(z)


class TruncatedNormalPosterior(WilsonBase):
    parameterization = 'structure_factor'
    def __init__(self, rasu, kl_weight, scale_factor=1e-1, eps=1e-12, low=1e-32, high=1e4, **kwargs):
        super().__init__(rasu, kl_weight=kl_weight, **kwargs)
        self.prior = WilsonPrior(
            rasu.centric,
            rasu.epsilon,
        )

        self.rasu = rasu

        self.loc = tfu.TransformedVariable(
            self.prior.mean(),
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.scale = tfu.TransformedVariable(
            scale_factor * self.prior.stddev(),
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )
        self.epsilon = eps
        self.low = low
        self.high = high

    @property
    def flat_distribution(self):
        low = tf.where(self.rasu.centric, 0., self.epsilon)
        return TruncatedNormal(self.loc, self.scale, low, self.high)

    def params(self, hkl):
        loc = self.rasu.gather(self.loc, hkl)
        scale = self.rasu.gather(self.scale, hkl)
        centric = self.rasu.gather(self.centric, hkl)
        low = tf.where(self.rasu.centric, 0., self.epsilon)
        high = self.high
        return loc, scale, low, high

    def distribution(self, hkl):
        params = self.params(hkl)
        return TruncatedNormal(*params)

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

