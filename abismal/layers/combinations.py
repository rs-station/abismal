import tensorflow as tf
import tf_keras as tfk

"""
Various models that form convex combinations of inputs.
"""

class ConvexCombination(tfk.layers.Layer):
    def __init__(self, kernel_initializer='glorot_normal', dropout=None):
        super().__init__()
        if dropout is not None:
            self.dropout = tfk.layers.Dropout(dropout)
        else:
            self.dropout = None
        self.softmax = tfk.layers.Softmax(axis=-1)
        self.linear = tfk.layers.Dense(1, kernel_initializer=kernel_initializer, use_bias=False)

    def call(self, data, mask=None, encoding=None, **kwargs):
        if encoding is None:
            encoding = data
        score = self.linear(data)
        if mask is None:
            mask = tf.ones_like(score)
        if self.dropout is not None:
            mask = self.dropout(mask)
        score = tf.squeeze(score, axis=-1)
        score = self.softmax(score, mask=tf.squeeze(mask, axis=-1))
        encoded = tf.reduce_sum(score[...,None] * encoding, axis=-2, keepdims=True)
        return encoded

class ConvexCombinations(tfk.layers.Layer):
    def __init__(self, npoints, kernel_initializer='glorot_normal', dropout=None):
        super().__init__()
        if dropout is not None:
            self.dropout = tfk.layers.Dropout(dropout)
        else:
            self.dropout = None
        self.softmax = tfk.layers.Softmax(axis=-1)
        self.einsum = None
        self.npoints = npoints
        self.kernel_initializer = kernel_initializer

    def build(self, shape, **kwargs):
        self.einsum = tfk.layers.EinsumDense("...a,bac->...bc", kernel_initializer=self.kernel_initializer, output_shape=(self.npoints, 1), bias_axes='c')

    def call(self, data, mask=None, **kwargs):
        score = self.einsum(data)[...,0]
        if mask is None:
            mask = tf.ones_like(score)
        else:
            #TODO: is this correct??
            mask = np.ones_like(score) * mask[...,None,:]

        if self.dropout is not None:
            mask = self.dropout(mask)
        score = self.softmax(score, mask=mask)
        points = tf.einsum("...ab,...ac->...bc", score, data)
        return points

class Average(tfk.layers.Layer):
    def __init__(self, axis, keepdims=True):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def call(self, data, **kwargs):
        return tf.reduce_mean(data, self.axis, self.keepdims)
