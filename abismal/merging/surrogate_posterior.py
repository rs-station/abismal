import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.priors import WilsonPrior

class WilsonBase(tfk.models.Model):
    def __init__(self, rasu, **kwargs):
        super().__init__(**kwargs)
        self.rasu = rasu
        self.seen = self.add_weight(
            shape=self.rasu.asu_size,
            initializer='zeros',
            dtype='bool',
            trainable=False,
            name="hkl_tracker",
        )

    def _to_dataset(self):
        raise NotImplementedError("Subclasses must implement _to_dataset")

    def _register_seen(self, hkl):
        unique,_ = tf.unique(tf.reshape(self.rasu._miller_ids(hkl), [-1]))
        unique = unique[unique!=-1]
        seen_batch = tf.scatter_nd(unique[:,None], tf.ones_like(unique, dtype='bool'), shape=[self.rasu.asu_size])
        self.seen.assign(self.seen | seen_batch)

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
        return out


class PosteriorCollection(tfk.models.Model):
    """ A collection of Wilson Posteriors """
    def __init__(self, *posteriors):
        super().__init__()
        self.posteriors = posteriors

    def _wp_method_helper(self, asu_id, hkl, value_func):
        out = None
        for i, wp in enumerate(self.posteriors):
            vals = value_func(hkl, wp)
            if out is None:
                out = vals
            else:
                aidx = asu_id == i
                out = tf.where(
                    aidx, 
                    vals,
                    out,
                )
        return out

    def mean(self, asu_id, hkl, **kwargs):
        out = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.mean(h),
        )
        return out

    def stddev(self, asu_id, hkl, **kwargs):
        out = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.stddev(h),
        )
        return out

    def call(self, asu_id, hkl, training=None):
        conc = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.rasu.gather(wp.concentration, h),
        )
        rate = self._wp_method_helper(
            asu_id,
            hkl,
            lambda h,wp: wp.rasu.gather(wp.rate, h),
        )
        for i,wp in enumerate(self.posteriors):
            _hkl = hkl[tf.squeeze(asu_id, axis=-1) == i] #srsly? y squeeze?
            if training:
                wp._register_seen(_hkl)
                wp.kl_div(_hkl, training=training)

        q = tfd.Gamma(conc, rate)
        return q

def WilsonPrior(centric, epsilon, sigma=1.):
    concentration = tf.where(centric, 0.5, 1.)
    rate = tf.where(centric, 0.5 / sigma / epsilon, 1. / sigma / epsilon)
    return tfd.Gamma(concentration, rate)

class WilsonPosterior(WilsonBase):
    def __init__(self, rasu, kl_weight, scale_factor=1e-2, eps=1e-12, **kwargs):
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
                #tfb.Shift(1. + eps), 
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

    @property
    def flat_distribution(self):
        return tfd.Gamma(self.concentration, self.rate)

    def distribution(self, hkl):
        conc = self.rasu.gather(self.concentration, hkl)
        rate = self.rasu.gather(self.rate, hkl)
        return tfd.Gamma(conc, rate)

    def _to_dataset(self):
        h,k,l = self.rasu.Hunique.T
        q = self.flat_distribution
        I = q.mean()      
        SIGI = q.stddev()
        out = rs.DataSet({
            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
            'I' : rs.DataSeries(I, dtype='J'),
            'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
            },
            merged=True,
            cell=self.rasu.cell,
            spacegroup=self.rasu.spacegroup,
        )
        return out

    def mean(self, hkl):
        q = self.flat_distribution
        mean = self.rasu.gather(q.mean(), hkl)
        return mean

    def stddev(self, hkl):
        q = self.flat_distribution
        stddev = self.rasu.gather(q.stddev(), hkl)
        return stddev

    def kl_div(self, hkl=None, training=None):
        q,p = self.flat_distribution, self.prior
        kl_div = q.kl_divergence(p)
        kl_div = tf.reduce_mean(kl_div)
        if training:
            self.add_metric(kl_div, name='KL')
            self.add_loss(self.kl_weight * kl_div)
        return kl_div

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

