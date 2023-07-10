import tensorflow as tf
import numpy as np
from tensorflow import keras as tfk
import tensorflow_probability as tfp



class Normalization(tfk.layers.Layer):
    """ 
    Fancier batch normalization based on the TFP implementation of the Welford algorithm for 
    online variance estimation. 
    """
    def __init__(self,  eps=1e-6, momentum=0.999, countmax=None, center=True):
        super().__init__()
        self._initialized = False
        self.eps = eps
        self.momentum = momentum
        self.countmax = countmax
        self.center = center

    def build(self, shapes, **kwargs):
        size = shapes[-1]
        shape = (size,)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros", trainable=False, dtype='int32')
        self.loc = self.add_weight(name="loc", shape=shape, initializer="zeros", trainable=False)
        self.variance = self.add_weight(name="variance", shape=shape, initializer="ones", trainable=False)

    def call(self, data, training=False, **kwargs):
        if self.countmax is not None:
            if self.count > self.countmax:
                training = False

        if training:
            loc,variance = tfp.stats.assign_moving_mean_variance(
                data,
                self.loc,
                moving_variance=self.variance,
                zero_debias_count=self.count,
                decay=self.momentum,
                axis=-2,
            )
        if self.center:
            return (data - self.loc) / tf.math.sqrt(self.variance + self.eps)
        return data / tf.math.sqrt(self.variance + self.eps)


