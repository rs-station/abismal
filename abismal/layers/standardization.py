import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import stats as tfs


@tfk.saving.register_keras_serializable(package="abismal")
class Standardize(tfk.layers.Layer):
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

    def standardize(self, data):
        mean,var = self._debiased_mean_variance()
        std = tf.clip_by_value(tf.sqrt(var), self.epsilon, np.inf)
        if self.center:
            return (data - mean) / std
        return data / std

    def call(self, data, training=None):
        if training and self.trainable:
            self.update(data)
        return self.standardize(data)



