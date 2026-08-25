import tensorflow as tf
import tf_keras as tfk


class FeedForward(tfk.layers.Layer):
    """
    This is a ResNet version 2 style layer
    """

    norm_dict = {
        "layer": lambda s, x: (x - tf.math.reduce_mean(x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + s.epsilon),
        "l2": lambda s, x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + s.epsilon * s.epsilon),
        "rms": lambda s, x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + s.epsilon * s.epsilon),
        "batch": lambda s, x: (x - tf.math.reduce_mean(x, axis=-2, keepdims=True)) * tf.math.rsqrt(tf.math.reduce_variance(x, axis=-2, keepdims=True) + s.epsilon * s.epsilon),
        "batch_l2": lambda s, x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-2, keepdims=True) + s.epsilon * s.epsilon),
        "identity": lambda s, x: x,
    }

    @classmethod
    def get_transform(cls, name, layer):
        """
        Resolve `name` to a callable of one tensor.

        Names in `norm_dict` are normalizers; they need the layer itself, because
        they read its epsilon. Everything else goes to keras, so any named
        activation works in the same slot. `None` resolves to keras' linear
        activation, i.e. no transform.

        Parameters
        ----------
        name : str or callable or None
            A key of `norm_dict`, a keras activation name, or a callable.
        layer : FeedForward
            The layer the normalizer reads its epsilon from.

        Returns
        -------
        callable
        """
        if isinstance(name, str) and name in cls.norm_dict:
            normalizer = cls.norm_dict[name]
            return lambda x: normalizer(layer, x)
        return tfk.activations.get(name)

    def __init__(
        self,
        hidden_units=None,
        dropout=None,
        activation="ReLU",
        kernel_initializer="glorot_normal",
        pre_activation="rms",
        use_bias=False,
        epsilon=1e-3,
        scale_factor=None,
        **kwargs,
    ):
        """
        This is a ResNet version 2 style feedforward layer. It implements the following

        ```
        out = dropout(linear(activation(hidden_linear(pre_activation(in))))) + in
        ```
        Where dropout is optional.

        Parameters
        ----------
        hidden_units : int (optional)
            The size of the hidden layer. By default this will be 2 times the size of the input.
        dropout : float (optional)
            Apply dropout with this rate. Dropout occurs after the second linear layer. By default
            dropout is not used.
        activation : string or callable (optional)
            Applied between the two linear layers. Either a key of `norm_dict`, a
            string name of a keras activation, or a callable. The default is 'ReLU'.
        kernel_initializer : string or callable (optional)
            Either a string a keras intializer style function. The default is 'glorot_normal'.
        pre_activation : string or callable (optional)
            Applied to the input before the first linear layer. Takes the same values
            as `activation`, so this slot holds either a normalizer or an activation.
            The default is 'rms'.
        use_bias : bool (optional)
            Whether the dense layers include bias parameters. 
        epsilon : float (optional)
            The value of epsilon that is used in the denominator of some normalizers. 
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.kernel_initializer = kernel_initializer
        self.epsilon = epsilon
        self.scale_factor = scale_factor

        if dropout is not None:
            self.dropout = tfk.layers.Dropout(dropout)
        else:
            self.dropout = None

        # Resolved after epsilon is set: a normalizer closes over this layer.
        self.activation = self.get_transform(activation, self)
        self.pre_activation = self.get_transform(pre_activation, self)
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

    def call(self, X, **kwargs):
        out = X
        out = self.pre_activation(out)
        out = self.ff1(out)
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
        out = self.pre_activation(out)

        out = self.ff1(out) * self.activation(self.ff2(out))
        out = self.ff3(out)

        if self.dropout is not None:
            out = self.dropout(out)

        if self.scale_factor is not None:
            out = out * self.scale_factor

        out = out + X

        return out
