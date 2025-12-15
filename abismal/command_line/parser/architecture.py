title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

from abismal.layers import FeedForward


def int_or_none_type(x):
    if x.lower() == "none":
        return None
    x = int(x)
    if x <= 0:
        return None
    return x


args_and_kwargs = (
    (
        ("--d-model",),
        {
            "help": "The number of channels in the model with default 32.",
            "default": 32,
            "type": int,
        },
    ),
    (
        ("--layers",),
        {
            "help": "The number of feedfoward layers with default 20.",
            "default": 20,
            "type": int,
        },
    ),
    (
        ("--activation",),
        {
            "help": "The name of the activation function used in the scale model. The default is 'relu'",
            "default": "relu",
            "type": str,
        },
    ),
    (
        ("--normalizer",),
        {
            "help": "Optional pre-normalization function for feed forward layers. The default is 'rms'",
            "default": "rms",
            "type": str,
            "choices": FeedForward.norm_dict.keys(),
        },
    ),
    (
        ("--gated",),
        {
            "help": "Use a Gated (GLU) architecture for the feed forward layers.",
            "action": "store_true",
        },
    ),
    (
        ("--epsilon",),
        {
            "help": "A small constant for numerical stability.",
            "default": 1e-12,
            "type": float,
        },
    ),
    (
        ("--standardization-decay",),
        {
            "help": "Abismal uses Welford's algorithm to calculate the standard deviation of intensities. "
            "The estimates are updated based on an exponentially decaying average. "
            "This setting controls how quickly the exponential falls off and the model 'forgets' previous observations. "
            "The default is 0.999 which corresponds to about 1000 steps for the decay half life."
            "The moements estimated by abismal are frozen after the first training epoch.",
            "default": 0.999,
            "type": float,
        },
    ),
)
