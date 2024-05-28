import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import stats as tfs


class RunningMoments:
    def __init__(self, axis=-2):
        self.n=0
        self.s=0
        self.mean=0
        self.axis=axis

    def update(self, x):
        k = len(x)
        self.n += k
        diff = x - self.mean
        self.mean = self.mean + np.sum(diff / self.n, axis=self.axis, keepdims=True)
        diff *= (x - self.mean)
        self.s = self.s + diff.sum(axis=self.axis, keepdims=True)

    @property
    def var(self):
        if self.n <= 1:
            return None
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt(self.var)

class Standardize(tfk.layers.Layer):
    def __init__(self, center=True, max_counts=np.inf, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts

    def build(self, shape):
        mom_shape = [1] * len(shape)
        mom_shape[-1] = shape[-1]
        reduce_dims = list(range(len(shape)-1))
        self.reduce_dims = reduce_dims
        self.axis_size = shape[-1]

        self.mean = self.add_weight(
            shape=mom_shape,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self.m2= self.add_weight(
            shape=mom_shape,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self.count = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='count',
        )

    @property
    def count_float(self):
        return tf.cast(self.count, self.mean.dtype)

    @property
    def std(self):
        return tf.sqrt(self.var)

    @property
    def var(self):
        m2 = tf.clip_by_value(self.m2, self.epsilon, np.inf)
        return m2 / self.count_float

    def update(self, x):
        k = tf.reduce_sum(tf.ones_like(x)) / self.axis_size
        self.count.assign_add(k)
        diff = x - self.mean
        new_mean = self.mean + \
            tf.reduce_sum(diff, axis=self.reduce_dims, keepdims=True) / self.count
        self.mean.assign(new_mean)
        diff *= (x - self.mean)
        self.m2.assign(
            self.m2 + tf.reduce_sum(diff, axis=self.reduce_dims, keepdims=True)
        )

    def standardize(self, data):
        if self.center:
            return (data - self.mean) / self.std
        return data / self.std

    def call(self, data, training=True):
        if self.max_counts > self.max_counts:
             training = False
        if training:
            self.update(data)
        return self.standardize(data)



if __name__=="__main__":
    l = 10_000
    d = 5
    x = np.arange(l)[...,None] * np.ones((l, d))
    x = x.astype('float32')
    n = RunningMoments()
    s = Standardize()
    for batch in np.split(x, 10):
        n.update(batch)
        s(batch)
        assert np.allclose(n.mean, s.mean)
        assert np.allclose(n.var, s.var)
        assert np.allclose(n.std, s.std)

