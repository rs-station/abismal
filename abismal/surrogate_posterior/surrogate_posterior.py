import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk

class PosteriorBase(tfk.models.Model):
    def __init__(self, rac, epsilon=1e-12, kl_weight=1., **kwargs):
        """
        rac : ReciprocalASUCollection
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.kl_weight = kl_weight
        self.rac = rac
        self.seen = self.add_weight(
            shape=self.rac.asu_size,
            initializer='zeros',
            dtype='bool',
            trainable=False,
            name="hkl_tracker",
        )

    def register_seen(self, asu_id, hkl):
        unique,_ = tf.unique(tf.reshape(self.rac._miller_ids(asu_id, hkl), [-1]))
        unique = unique[unique!=-1]
        seen_batch = tf.scatter_nd(
            unique[:,None], 
            tf.ones_like(unique, dtype='bool'), 
            shape=[self.rac.asu_size]
        )
        self.seen.assign(self.seen | seen_batch)

    def flat_distribution(self):
        raise NotImplementedError("Subclasses must implement a flat_distribution method")

    def flat_prior(self):
        raise NotImplementedError("Subclasses must implement a flat_prior method")

    def register_kl(self, ipred=None, asu_id=None, hkl=None, training=None):
        raise NotImplementedError("Subclasses must implement a register_kl method")

    def to_datasets(self, seen=True):
        """
        Parameters
        ----------
        seen : bool (optional)
            Only include reflections seen during training. Defaults to True. 
        """
        raise NotImplementedError("Subclasses must implement a to_datasets method")

    def mean(self, asu_id, hkl):
        q = self.flat_distribution
        mean = self.rac.gather(q.mean(), asu_id, hkl)
        return mean

    def stddev(self, asu_id, hkl):
        q = self.flat_distribution
        stddev = self.rac.gather(q.stddev(), asu_id, hkl)
        return stddev

