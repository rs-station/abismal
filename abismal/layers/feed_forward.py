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
    norm_dict = {
        'rms' : lambda x: x / (tf.math.sqrt(tf.reduce_sum(x*x, axis=-1, keepdims=True)) + 1e-3),
        'layer' : lambda x: (x - tf.math.reduce_mean(x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-3),
        'identity' : lambda x: x,
    }
    def __init__(self, 
        dropout=None, 
        hidden_units=None, 
        activation='ReLU',
        kernel_initializer='glorot_normal', 
        normalizer='rms',
        use_bias=False,
        **kwargs
        ):
        """
        This is a ResNet version 2 style feedforward layer. It implements the following

        ```
        out = dropout(linear(activation(hidden_linear(activation(layer_norm(in)))))) + in
        ```
        Where dropout and layer normalization are optional. 

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
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.kernel_initializer = kernel_initializer
        self.normalizer = normalizer

        if dropout is not None:
            self.dropout = tfk.layers.Dropout(dropout)
        else:
            self.dropout = None

        self.activation = tfk.activations.get(activation)
        self.use_bias = use_bias

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
        out = self.norm_dict[self.normalizer](out)
        out = self.activation(out)
        out = self.ff1(out)
        out = self.norm_dict[self.normalizer](out)
        out = self.activation(out)
        out = self.ff2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + X
        return out


class GLUFeedForward(FeedForward):
    """
    This is a ResNet version 2 style layer
    """
    norm_dict = {
        'rms' : lambda x: x / (tf.math.sqrt(tf.reduce_sum(x*x, axis=-1, keepdims=True)) + 1e-3),
        'layer' : lambda x: (x - tf.math.reduce_mean(x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-3),
        'identity' : lambda x: x,
    }
    def build(self, shape, **kwargs):
        self.units = shape[-1]
        if self.hidden_units is None:
            self.hidden_units = 2 * self.units

        self.ff1 = tfk.layers.Dense(self.hidden_units, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, **kwargs)
        self.ff2 = tfk.layers.Dense(self.hidden_units, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, **kwargs)
        self.ff3 = tfk.layers.Dense(self.units, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, **kwargs)

        self.ff1.build(shape)
        self.ff2.build(shape)
        self.ff3.build(shape[:-1] + [self.hidden_units])

    def call(self, X, **kwargs):
        out = X
        out = self.norm_dict[self.normalizer](out)

        out = self.ff1(out) * self.activation(self.ff2(out))
        out = self.ff3(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + X

        return out

