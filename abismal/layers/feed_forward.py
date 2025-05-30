import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb


class FeedForward(tfk.layers.Layer):
    """
    This is a ResNet version 2 style layer
    """
    def __init__(self, 
        dropout=None, 
        hidden_units=None, 
        activation='ReLU',
        kernel_initializer='glorot_normal', 
        normalize=None, 
        epsilon=1e-3,
        **kwargs
        ):
        """

        Parameters
        ----------
        dropout : float (optional)
            Apply dropout with this rate. Dropout occurs after the second linear layer. By default
            dropout is not used.
        hidden_units : int (optional)
            The size of the hidden layer. By default this will be 2 times the size of the input.
        activation : string or callable (optional)
            Either a string name of a keras activation or a callable function. The default is 'ReLU'.
        kernel_initializer : string or callable (optional)
            Either a string a keras intializer style function. The default is 'glorot_normal'. 
        normalize : str (optional)
            Optionally apply normalization to the output. Options include 'layer' and 'rms'.
        epsilon : float (optional)
            If using normalization, this is a small constant added to the denominator for 
            numerical stability.
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.kernel_initializer = kernel_initializer
        self.epsilon = epsilon

        if dropout is not None:
            self.dropout = tfk.layers.Dropout(dropout)
        else:
            self.dropout = None

        self.normalize = normalize
        self.activation = tfk.activations.get(activation)
        self.use_bias = True

    def rms_normalize(self, X):
        den = tf.square(X)
        den = tf.math.reduce_sum(den, -1, keepdims=True)
        out = tf.sqrt(den + self.epsilon)
        out = X / den
        return out

    def layer_normalize(self, X):
        loc = tf.math.reduce_mean(X, axis=-1, keepdims=True)
        scale = tf.math.reduce_std(X, axis=-1, keepdims=True) + self.epsilon
        return (X - loc) / scale

    def build(self, shape, **kwargs):
        self.units = shape[-1]
        if self.hidden_units is None:
            self.hidden_units = 2 * self.units

        self.ff1 = tfk.layers.Dense(self.hidden_units, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, **kwargs)
        self.ff2 = tfk.layers.Dense(self.units, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, **kwargs)

        self.ff1.build(shape)
        self.ff2.build(shape[:-1] + [self.hidden_units])

    def call(self, X, **kwargs):
        out = X
        out = self.activation(out)
        out = self.ff1(out) 
        out = self.activation(out)
        out = self.ff2(out)
        out = out + X
        if self.normalize == 'layer':
            out = self.layer_normalize(out)
        if self.normalize == 'rms':
            out = self.rms_normalize(out)
        return out

#        if self.dropout is not None:
#            out = self.dropout(out)
#
#        out = out + X
#        return out
#
#
#        if self.normalize is not None:
#            out = self.layer_normalize(out)
#        out = self.activation(out)
#        out = self.ff1(out)
#        if self.normalize is not None:
#            out = self.layer_normalize(out)
#        out = self.activation(out)
#        out = self.ff2(out)
#
#        if self.dropout is not None:
#            out = self.dropout(out)
#
#        out = out + X
#        return out
#
