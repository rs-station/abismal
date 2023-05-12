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

    @property
    def flat_distribution(self):
        raise NotImplementedError("Subclasses must implement a flat_distribution property")

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


class PosteriorCollectionBase(tfk.models.Model):
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
        raise NotImplementedError()

