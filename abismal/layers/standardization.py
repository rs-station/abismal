import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import stats as tfs


class Standardize(tfk.layers.Layer):
    def __init__(self, center=True, decay=0.999, epsilon=1e-3):
        super().__init__()
        self.decay = decay
        self.center = center
        self.epsilon = epsilon

    def build(self, shape):
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
            dtype=tf.int32,
            trainable=False,
            name='zero_debias_count',
        )

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
            axis=-2,
        )

    def standardize(self, data):
        mean,var = self._debiased_mean_variance()
        std = tf.clip_by_value(tf.sqrt(var), self.epsilon, np.inf)
        if self.center:
            return (data - mean) / std
        return data / std

    def call(self, data, training=True):
        if training:
            self.update(data)
        return self.standardize(data)



