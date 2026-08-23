title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

from abismal.layers import FeedForward
from abismal.command_line.parser.custom_types import transform_name


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
            "help": "The number of channels in the model with default 256.",
            "default": 256,
            "type": int,
        },
    ),
    (
        ("--layers",),
        {
            "help": "The number of feedfoward layers with default 5.",
            "default": 5,
            "type": int,
        },
    ),
    (
        ("--activation",),
        {
            "help": "Applied between the two linear layers of each feed forward layer. "
            "Either a normalizer -- one of " + ", ".join(sorted(FeedForward.norm_dict))
            + " -- or any keras activation. The default is 'relu'.",
            "default": "relu",
            "type": transform_name,
        },
    ),
    (
        ("--pre-activation",),
        {
            "help": "Applied to each feed forward layer's input, before the first linear "
            "layer. Takes the same values as --activation. The default is 'relu'.",
            "default": "relu",
            "type": transform_name,
        },
    ),
    (
        ("--batch-normalize",),
        {
            "help": "Use normalize the scale model inputs.",
            "action": "store_true",
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
        ("--metadata-noise-factor",),
        {
            "help": "Standard deviation of the Gaussian noise added to the metadata during training. "
            "This regularizes the model against overfitting on small data sets. The default is 0.1.",
            "default": 0.1,
            "type": float,
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
        ("--ff-epsilon",),
        {
            "help": "Epsilon used by the feed forward layers' --pre-activation, when that "
            "is a normalizer. Defaults to 0.0.",
            "default": 0.0,
            "type": float,
        },
    ),
    (
        ("--dropout",),
        {
            "help": "Apply dropout when pooling reflections. ",
            "default": None,
            "type": float,
        },
    ),
)
