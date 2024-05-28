import numpy as np
import tensorflow as tf
import tf_keras as tfk

class ScaleRange(tfk.layers.Layer):
    def __init__(self, minimum=-1, maximum=1, axis=-1):
        super().__init__()
        self.axis = axis
        self.freeze = False
        self.range = maximum - minimum
        self.min = minimum

    def build(self, shape):
        param_shape = [1 for i in shape]
        param_shape[self.axis] = shape[self.axis]
        self.reduce_dims = list(range(len(shape)))
        del self.reduce_dims[self.axis]
        self.low = self.add_weight(
            shape=param_shape,
            initializer=tfk.initializers.Constant(np.inf),
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self.high = self.add_weight(
            shape=param_shape,
            initializer=tfk.initializers.Constant(-np.inf),
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )

    def call(self, data, training=None):
        if training:
            data_min = tf.reduce_min(data, axis=self.reduce_dims, keepdims=True)
            data_max = tf.reduce_max(data, axis=self.reduce_dims, keepdims=True)
            self.low.assign(tf.minimum(data_min, self.low))
            self.high.assign(tf.maximum(data_max, self.high))

        out = self.range * (data - self.low) / (self.high - self.low) + self.min
        return out

class PositionalEncoding(tfk.layers.Layer):
    """ Positionally encode the last axis of an array. """
    def __init__(self, num_frequencies):
        super().__init__()
        self.scale = ScaleRange(axis=-1)
        self.num_frequencies = num_frequencies

    def call(self, data, training=None):
        p = self.scale(data, training=training)
        d = p.get_shape()[-1]
        p = tf.repeat(p, self.num_frequencies, axis=-1)

        L = tf.range(0, self.num_frequencies, dtype='float32')
        f = np.pi * 2 ** L
        f = tf.tile(f, (d,))
        fp = f * p

        out = tf.concat((
            tf.sin(fp),
            tf.cos(fp),
        ), axis=-1)
        return out


if __name__=='__main__':
    data = np.random.random((100, 37, 10))
    pe = PositionalEncoding(4)
    out = pe(data, training=True)
    print(out.shape)

