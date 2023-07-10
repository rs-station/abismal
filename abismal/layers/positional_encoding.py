import tensorflow as tf
import numpy as np
from tensorflow import keras as tfk


class Normalization(tfk.layers.Layer):
    """ Normalize the last axis online between -1 and 1 """ 
    def __init__(self, max_initializer='ones', min_initializer='zeros'):
        super().__init__()
        self.min=None
        self.max=None
        self._max_initializer=max_initializer
        self._min_initializer=min_initializer

    def build(self, input_shape):
       self.min = self.add_weight(
            shape=input_shape[-1],
            initializer=self._min_initializer,
            trainable=False,
        )
       self.max = self.add_weight(
            shape=input_shape[-1],
            initializer=self._max_initializer,
            trainable=False,
        )

    def call(self, X):
        ndim = len(X.shape)
        batch_min = tf.reduce_min(X, tf.range(-ndim, -1))
        batch_max = tf.reduce_max(X, tf.range(-ndim, -1))

        self.min.assign(tf.minimum(batch_min, self.min))
        self.max.assign(tf.maximum(batch_max, self.max))

        return 2.*(X - self.min) / (self.max - self.min) - 1.

class PositionalEncoding(tfk.layers.Layer):
    def __init__(self,  L, active_dims=None, *args, **kwargs):
        """
        L : int
            Bit depth of positional encoding
        """
        super().__init__(*args, **kwargs)
        self.L = L
        self.active_dims = active_dims

    def call(self, Q):
        d = Q.shape[-1]
        active_dims = self.active_dims if self.active_dims is not None else range(d)
        to_encode = tf.gather(Q, active_dims, axis=-1)
        cos = [tf.math.cos(to_encode*np.pi*2**f) for f in range(self.L)]
        sin = [tf.math.sin(to_encode*np.pi*2**f) for f in range(self.L)]
        return tf.concat([Q] + cos + sin, -1)
