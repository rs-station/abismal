import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import stats as tfs


@tfk.saving.register_keras_serializable(package="abismal")
class StandardizeBase(tfk.layers.Layer):
    def __init__(self, center=True, decay=0.999, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.decay = decay
        self.center = center
        self.epsilon = epsilon
        self.count = None

    def build(self, shape):
        #Check if already built
        if self.count is not None:
            return
        d = shape[-1]

        self._mean = self.add_weight(
            shape=d,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self._var = self.add_weight(
            shape=d,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='variance',
        )
        self.count = self.add_weight(
            shape = (),
            initializer = 'zeros',
            dtype=tf.int64,
            trainable=False,
            name='zero_debias_count',
        )

    def get_config(self):
        conf = super().get_config()
        conf.update({
            'center' : self.center,
            'decay' : self.decay,
            'epsilon' : self.epsilon,
        })
        return conf

    def _debiased_mean_variance(self):
        mean,var = tfs.moving_mean_variance_zero_debiased(
            self._mean,
            self._var,
            self.count,
            decay=self.decay,
        )
        return mean,var

    @property
    def mean(self):
        mean,_ = self._debiased_mean_variance()
        return mean

    @property
    def var(self):
        _,var = self._debiased_mean_variance()
        return var

    @property
    def std(self):
        s = tf.sqrt(self.var)
        return tf.clip_by_value(s, self.epsilon, np.inf)

    def update(self, x):
        tfs.assign_moving_mean_variance(
            x,
            self._mean,
            self._var,
            zero_debias_count=self.count,
            decay=self.decay,
            axis=0, #TODO: if tf.rank(x) > 2, this should be (0, ... , tf.rank(x) - 2) i think
        )

    def standardize(self, data, **kwargs):
        mean,var = self._debiased_mean_variance()
        std = tf.clip_by_value(tf.sqrt(var), self.epsilon, np.inf)
        if self.center:
            return (data - mean) / std
        return data / std

class Standardize(StandardizeBase):
    def call(self, inputs, training=None, **kwwargs):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        if training and self.trainable:
            self.update(metadata.flat_values)
            #tf.ragged.map_flat_values(self.update, metadata)

        metadata = tf.ragged.map_flat_values(self.standardize, metadata)
        out = (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        )
        return out

class Normalize(Standardize):
    def normalize(self, data, **kwargs):
        mean,var = self._debiased_mean_variance()
        std = tf.clip_by_value(tf.sqrt(var), self.epsilon, np.inf)
        return data / mean 

    def call(self, inputs, training=None, **kwwargs):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        if training and self.trainable:
            self.update(iobs.flat_values)
            #tf.ragged.map_flat_values(self.update, metadata)

        iobs = tf.ragged.map_flat_values(self.normalize, iobs)
        sigiobs = tf.ragged.map_flat_values(self.normalize, sigiobs)
        out = (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        )
        return out

@tfk.saving.register_keras_serializable(package="abismal")
class BinnedNormalize(tfk.layers.Layer):
    def __init__(self, rac, bins=20, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.rac = rac
        self.bins = bins
        self.epsilon = epsilon
        self.mean = self.add_weight("mean", shape=bins, initializer='zeros', trainable=False)
        self.size = self.add_weight("size", shape=bins, initializer='zeros', trainable=False)
        from reciprocalspaceship.utils import bin_by_percentile
        self.labels,self.edges = bin_by_percentile(rac.dHKL, ascending=False)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'bins' : self.bins,
            'epsilon' : self.epsilon,
        })
        return config

    def update(self, data, idx):
        X = tf.squeeze(data, axis=-1)
        mask = tf.ones_like(X)

        batch_size = tf.scatter_nd(idx, mask, (self.bins,))
        batch_mean = tf.scatter_nd(idx, X, (self.bins,)) / batch_size

        self.size.assign(self.size + batch_size)
        self.mean.assign(self.mean + (batch_size / self.size) * (batch_mean - self.mean))

    def standardize(self, inputs, idx=None):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        if idx is None:
            idx = self.rac.gather(labels, asu_id, hkl_in)
        mean = tf.gather(self.mean, idx) + self.epsilon
        out = (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs / mean,
            sigiobs / mean,
        )
        return out

    def call(self, inputs, training=None):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        idx = self.rac.gather(self.labels, asu_id, hkl_in)[...,None]
        if training and self.trainable:
            self.update(iobs.flat_values, idx.flat_values)
            #tf.ragged.map_flat_values(self.update, iobs, idx)
        out = self.standardize(inputs, idx)
        return out

