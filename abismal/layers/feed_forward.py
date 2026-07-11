import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp


class FeedForward(tfk.layers.Layer):
    """
    This is a ResNet version 2 style layer
    """

    norm_dict = {
        "layer": lambda s, x: (x - tf.math.reduce_mean(x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + s.epsilon),
        "l2": lambda s, x: s.normalizer_gain * x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + s.epsilon * s.epsilon),
        "rms": lambda s, x: s.normalizer_gain * x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + s.epsilon * s.epsilon),
        "activation": lambda s, x: s.activation(x),
        "batch": lambda s, x: (x - tf.math.reduce_mean(x, axis=-2, keepdims=True)) * tf.math.rsqrt(tf.math.reduce_variance(x, axis=-2, keepdims=True) + s.epsilon * s.epsilon),
        "batch_l2": lambda s, x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-2, keepdims=True) + s.epsilon * s.epsilon),
        "identity": lambda s, x: s.normalizer_gain * x,
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
        scale_factor=None,
        normalizer_gain=0.1,
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
        normalizer_gain : float (optional)
            The gain setting that is used in the numerator of some normalizers. 
        """
        super().__init__()
        self.normalizer_gain = normalizer_gain
        self.hidden_units = hidden_units
        self.kernel_initializer = kernel_initializer
        self.normalizer = normalizer
        self.epsilon = epsilon
        self.scale_factor = scale_factor

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
            #kernel_initializer=self.kernel_initializer,
            kernel_initializer='zeros',
            use_bias=self.use_bias,
            **kwargs,
        )

        self.ff1.build(shape)
        self.ff2.build(shape[:-1] + [self.hidden_units])

    def normalize(self, X):
        return self.norm_dict[self.normalizer](self, X)

    def _track_input_norm(self, X):
        # Exact threshold for 'l2'; off by sqrt(d_model) for 'rms' (which thresholds
        # ||X||/sqrt(d_model) against epsilon instead).
        norm = tf.norm(X, axis=-1)
        # Fraction below epsilon: robust to a single outlier row, and answers the
        # actual question (how often does epsilon bind) rather than just the
        # worst case this batch.
        frac_below_eps = tf.reduce_mean(tf.cast(norm < self.epsilon, norm.dtype))
        self.add_metric(frac_below_eps, name=self.name + "_frac_below_eps")
        self.add_metric(tf.reduce_min(norm), name=self.name + "_x_norm_min")

    def _track_pre_activation(self, X):
        quants = tfp.stats.percentile(X, [0., 50., 100.])
        std = tf.math.reduce_std(X)
        mean= tf.math.reduce_mean(X)
        #self.add_metric(quants[0], self.name + '_pre_activation_min')
        #self.add_metric(quants[1], self.name + '_pre_activation_median')
        #self.add_metric(quants[2], self.name + '_pre_activation_max')
        self.add_metric(std, self.name + '_pre_activation_std')
        self.add_metric(mean, self.name + '_pre_activation_mean')

    def call(self, X, **kwargs):
        #self._track_input_norm(X)
        out = X
        #self._track_pre_activation(out)
        out = self.normalize(out)
        out = self.ff1(out)
        #self._track_pre_activation(out)
        out = self.activation(out)
        out = self.ff2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        if self.scale_factor is not None:
            out = out * self.scale_factor

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
            #kernel_initializer=self.kernel_initializer,
            kernel_initializer='zeros',
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

        if self.scale_factor is not None:
            out = out * self.scale_factor

        out = out + X

        return out
