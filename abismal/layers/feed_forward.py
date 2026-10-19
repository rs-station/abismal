import math
import torch
import torch.nn.functional as F


# Keras resolved activations by name; these are the ones the CLI offers (see
# command_line/parser/architecture.py), with keras' defaults where torch's
# differ: leaky_relu's slope is 0.2 in keras and 0.01 in torch, and keras'
# softmax runs over the last axis.
ACTIVATIONS = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.2),
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "tanh": torch.tanh,
    "elu": F.elu,
    "linear": lambda x: x,
}


def variance_scaling_(weight, scale=1.0, mode="fan_avg"):
    """
    In-place equivalent of keras' `VarianceScaling(scale, mode, "truncated_normal")`.

    Keras truncates at two standard deviations and divides the target standard
    deviation by 0.8796..., the standard deviation of a unit normal truncated
    there, so that the *truncated* draws end up with the requested variance.
    """
    fan_out, fan_in = weight.shape[0], weight.shape[1]
    fan = {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "fan_avg": 0.5 * (fan_in + fan_out),
    }[mode]
    std = math.sqrt(scale / fan) / 0.87962566103423978
    with torch.no_grad():
        return torch.nn.init.trunc_normal_(weight, 0.0, std, -2.0 * std, 2.0 * std)


def glorot_normal_(weight):
    return variance_scaling_(weight, 1.0, "fan_avg")


def linear(in_features, out_features, use_bias=True, kernel_init=glorot_normal_):
    """A `torch.nn.Linear` initialized the way keras initializes a `Dense`."""
    layer = torch.nn.Linear(in_features, out_features, bias=use_bias)
    if kernel_init == "zeros":
        torch.nn.init.zeros_(layer.weight)
    else:
        kernel_init(layer.weight)
    if use_bias:
        torch.nn.init.zeros_(layer.bias)
    return layer
class FeedForward(torch.nn.Module):
    """
    This is a ResNet version 2 style layer
    """

    norm_dict = {
        "layer": lambda s, x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, correction=0) + s.epsilon),
        "l2": lambda s, x: x * torch.rsqrt(x.square().sum(-1, keepdim=True) + s.epsilon * s.epsilon),
        "rms": lambda s, x: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + s.epsilon * s.epsilon),
        "batch": lambda s, x: (x - x.mean(-2, keepdim=True)) * torch.rsqrt(x.var(-2, keepdim=True, correction=0) + s.epsilon * s.epsilon),
        "batch_l2": lambda s, x: x * torch.rsqrt(x.square().sum(-2, keepdim=True) + s.epsilon * s.epsilon),
        "identity": lambda s, x: x,
    }

    @classmethod
    def get_transform(cls, name, layer):
        """
        Resolve `name` to a callable of one tensor.

        Names in `norm_dict` are normalizers; they need the layer itself, because
        they read its epsilon. Everything else is a named activation, so any
        activation works in the same slot. `None` resolves to the identity.

        Parameters
        ----------
        name : str or callable or None
            A key of `norm_dict`, an activation name, or a callable.
        layer : FeedForward
            The layer the normalizer reads its epsilon from.

        Returns
        -------
        callable
        """
        if name is None:
            return ACTIVATIONS["linear"]
        if callable(name):
            return name
        if name in cls.norm_dict:
            normalizer = cls.norm_dict[name]
            return lambda x: normalizer(layer, x)
        key = name.lower()
        if key not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {name!r}. Choose from {sorted(ACTIVATIONS)} or {sorted(cls.norm_dict)}")
        return ACTIVATIONS[key]

    def __init__(
        self,
        units,
        hidden_units=None,
        dropout=None,
        activation="ReLU",
        kernel_initializer=glorot_normal_,
        pre_activation="rms",
        use_bias=False,
        epsilon=1e-3,
        scale_factor=None,
    ):
        """
        This is a ResNet version 2 style feedforward layer. It implements the following

        ```
        out = dropout(linear(activation(hidden_linear(pre_activation(in))))) + in
        ```
        Where dropout is optional.

        Parameters
        ----------
        units : int
            The width of the input and output.
        hidden_units : int (optional)
            The size of the hidden layer. By default this will be 2 times the size of the input.
        dropout : float (optional)
            Apply dropout with this rate. Dropout occurs after the second linear layer. By default
            dropout is not used.
        activation : string or callable (optional)
            Applied between the two linear layers. Either a key of `norm_dict`, the
            name of an activation, or a callable. The default is 'ReLU'.
        kernel_initializer : callable (optional)
            An in-place initializer for the first linear layer's weight. The
            default is keras' 'glorot_normal'.
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
        self.units = units
        self.hidden_units = 2 * units if hidden_units is None else hidden_units
        self.epsilon = epsilon
        self.scale_factor = scale_factor
        self.use_bias = use_bias
        self.dropout_rate = dropout
        self._activation = activation
        self._pre_activation = pre_activation

        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        # Resolved after epsilon is set: a normalizer closes over this layer.
        self.activation = self.get_transform(activation, self)
        self.pre_activation = self.get_transform(pre_activation, self)
        self.build(kernel_initializer)

    def build(self, kernel_initializer):
        self.ff1 = linear(self.units, self.hidden_units, self.use_bias, kernel_initializer)
        self.ff2 = linear(self.hidden_units, self.units, self.use_bias, "zeros")

    def forward(self, X):
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

    def build(self, kernel_initializer):
        self.ff1 = linear(self.units, self.hidden_units, self.use_bias, kernel_initializer)
        self.ff2 = linear(self.units, self.hidden_units, self.use_bias, kernel_initializer)
        self.ff3 = linear(self.hidden_units, self.units, self.use_bias, "zeros")

    def forward(self, X):
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
