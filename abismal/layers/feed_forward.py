import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb


class FeedForward(tfk.layers.Layer):
    """
    This is a ResNet version 2 style layer
    """

    norm_dict = {
        "layer": lambda s, x: (x - tf.math.reduce_mean(x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + s.epsilon),
        "rms": lambda s, x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + s.epsilon * s.epsilon),
        "activation": lambda s, x: s.activation(x),
    }

    def __init__(
        self,
        hidden_units=None,
        dropout=None,
        activation="ReLU",
        kernel_initializer="glorot_normal",
        normalizer="rms",
        use_bias=False,
        epsilon=1e-3,
        **kwargs,
    ):
        """
        This is a ResNet version 2 style feedforward layer. It implements the following

        ```
        out = dropout(linear(activation(hidden_linear(activation(layer_norm(in)))))) + in
        ```
        Where dropout and layer normalization are optional.

        Parameters
        ----------
        hidden_units : int (optional)
            The size of the hidden layer. By default this will be 2 times the size of the input.
        dropout : float (optional)
            Apply dropout with this rate. Dropout occurs after the second linear layer. By default
            dropout is not used.
        activation : string or callable (optional)
            Either a string name of a keras activation or a callable function. The default is 'ReLU'.
        kernel_initializer : string or callable (optional)
            Either a string a keras intializer style function. The default is 'glorot_normal'.
        normalizer : string (optional)
            The type of normalization to use. 
        use_bias : bool (optional)
            Whether the dense layers include bias parameters. 
        epsilon : float (optional)
            The value of epsilon that is used in the denominator of some normalizers. 
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.kernel_initializer = kernel_initializer
        self.normalizer = normalizer
        self.epsilon = epsilon

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

        self.ff1 = tfk.layers.Dense(
            self.hidden_units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            **kwargs,
        )
        self.ff2 = tfk.layers.Dense(
            self.units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            **kwargs,
        )

        self.ff1.build(shape)
        self.ff2.build(shape[:-1] + [self.hidden_units])

    def normalize(self, X):
        return self.norm_dict[self.normalizer](self, X)

    def call(self, X, **kwargs):
        out = X
        out = self.normalize(out)
        out = self.ff1(out)
        out = self.activation(out)
        out = self.ff2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + X
        return out


class GLUFeedForward(FeedForward):
    """
    This is a residual, gated linear unit. 
    """

    def build(self, shape, **kwargs):
        self.units = shape[-1]
        if self.hidden_units is None:
            self.hidden_units = 2 * self.units

        self.ff1 = tfk.layers.Dense(
            self.hidden_units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            **kwargs,
        )
        self.ff2 = tfk.layers.Dense(
            self.hidden_units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            **kwargs,
        )
        self.ff3 = tfk.layers.Dense(
            self.units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            **kwargs,
        )

        self.ff1.build(shape)
        self.ff2.build(shape)
        self.ff3.build(shape[:-1] + [self.hidden_units])

    def call(self, X, **kwargs):
        out = X
        out = self.normalize(out)

        out = self.ff1(out) * self.activation(self.ff2(out))
        out = self.ff3(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + X

        return out
