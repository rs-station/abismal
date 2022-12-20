import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from reciprocal_asu import ReciprocalASU
from abismal_zero.layers import *
from abismal_zero.blocks import *
from abismal_zero.priors import WilsonPrior

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

class WilsonPosterior(WilsonBase):
    def __init__(self, rasu, kl_weight, scale_factor=1e-2, eps=1e-12, **kwargs):
        super().__init__(rasu, **kwargs)
        self.prior = WilsonPrior(
            rasu.centric,
            rasu.epsilon,
        )

        self.rasu = rasu
        infinity = 1e10
        zero = tf.where(rasu.centric, 0., 1e-32)
        self.kl_weight = kl_weight
        loc = tfu.TransformedVariable(
            self.prior.mean(),
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )

        scale = tfu.TransformedVariable(
            self.prior.stddev() * scale_factor,
            tfb.Chain([
                tfb.Shift(eps), 
                #tfb.Softplus(),
                tfb.Exp(),
            ]),
        )
        self.flat_distribution = tfd.TruncatedNormal(loc, scale, zero, infinity)

    def _to_dataset(self):
        h,k,l = self.rasu.Hunique.T
        F = self.flat_distribution.mean()      
        SIGF = self.flat_distribution.stddev()
        out = rs.DataSet({
            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
            'F' : rs.DataSeries(F, dtype='F'),
            'SIGF' : rs.DataSeries(SIGF, dtype='Q'),
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

    def call(self, hkl, mc_samples=1, mask=None, training=None):
        q,p = self.flat_distribution, self.prior
        z = q.sample(mc_samples)
        z = tf.maximum(q.low, z)

        q_z = q.log_prob(z)
        p_z = p.log_prob(z)

        F = self.rasu.gather(tf.transpose(z), hkl)
        I = tf.square(F)

        #TODO: which kl version is better? the one that resamples based on the batch millers or the one that just uses all the samples?
        # furthermore -- would it be better to use the unique set of miller indices for a batch??
        kl_div = q_z - p_z

        # Don't do any accounting for multiplicity or batch millers
        #kl_div = tf.reduce_mean(kl_div)

        # Only regularize observed millers on each batch (might imply a different value for kl_weight)
        unique,_ = tf.unique(tf.reshape(self.rasu._miller_ids(hkl), [-1]))
        count = mc_samples * tf.reduce_sum(tf.where(unique == -1, 0., 1.)) 
        kl_div = tf.reduce_sum(tf.gather(kl_div, unique, axis=-1)) / count

        # Resample from batch miller indices
        #kl_div = self.rasu.gather(kl_div, hkl)[:,:,None] 
        #kl_div = tf.reduce_sum(tf.where(mask==1, kl_div, 0.)) / tf.reduce_sum(mask)
        #kl_div = tf.reduce_mean(kl_div) 

        self.add_metric(kl_div, name='KL')
        self.add_loss(self.kl_weight * kl_div)

        if training:
            self._register_seen(hkl)

        return I

