title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

from abismal.layers import FeedForward

# The activations either of the two transform slots may be given. These are the
# names layers/feed_forward.py maps onto torch functions; listing them as
# choices makes the CLI reject anything else up front, and makes the option
# render as a dropdown in the notebook GUI, which builds itself from
# `action.choices`.
NAMED_ACTIVATIONS = (
    "relu",
    "swish",
    "leaky_relu",
    "gelu",
    "sigmoid",
    "softmax",
    "tanh",
    "elu",
)

# The normalizers come from the layer, so adding one to `norm_dict` reaches the CLI
# with no change here.
TRANSFORM_CHOICES = tuple(sorted(FeedForward.norm_dict)) + NAMED_ACTIVATIONS


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
            "Either a normalizer or a named activation. The default is 'relu'.",
            "default": "relu",
            "choices": TRANSFORM_CHOICES,
        },
    ),
    (
        ("--pre-activation",),
        {
            "help": "Applied to each feed forward layer's input, before the first linear "
            "layer. Takes the same values as --activation. The default is 'relu'.",
            "default": "relu",
            "choices": TRANSFORM_CHOICES,
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
        ("--deep-context",),
        {
            "help": "Pool the image track after every layer (leave-one-out) into the scale track, ahead "
            "of the layer of the same depth. 'projected' (the default when the flag is given) adds a "
            "learned projection of the pooled residual stream, 'raw' the pooled residual stream itself, "
            "'delta' the pooled increment of each layer. Without the flag the image track is pooled "
            "once, at the end, into the scale track's input.",
            "nargs": "?",
            "const": "projected",
            "default": None,
            "choices": ["projected", "raw", "delta"],
        },
    ),
    (
        ("--posterior-features",),
        {
            "help": "Give the scale model's pooling branch the current posterior mean and "
            "standard deviation of each reflection's structure factor (no gradient).",
            "action": "store_true",
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
