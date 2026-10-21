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
            "help": "The number of channels in the model with default 128.",
            "default": 128,
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
            "Either a normalizer or a named activation. The default is 'swish', which "
            "with the gated layers makes them SwiGLU.",
            "default": "swish",
            "choices": TRANSFORM_CHOICES,
        },
    ),
    (
        ("--pre-activation",),
        {
            "help": "Applied to each feed forward layer's input, before the first linear "
            "layer. Takes the same values as --activation. The default is 'rms', "
            "RMS normalization.",
            "default": "rms",
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
        ("--no-gated",),
        {
            "help": "Use plain feed forward layers instead of the gated (GLU) ones, "
            "which are the default.",
            "action": "store_false",
            "dest": "gated",
        },
    ),
    (
        ("--hidden-multiplier",),
        {
            "help": "Width of each feed forward layer's hidden layer, as a multiple of "
            "--d-model. The default, 2/3, gives a gated layer the parameters of a plain "
            "one with twice --d-model hidden units.",
            "default": 2 / 3,
            "type": float,
        },
    ),
    (
        ("--no-posterior-features",),
        {
            "help": "Do not give the scale model's pooled image representation the current "
            "posterior mean and standard deviation of each reflection (without gradient), "
            "which it gets by default.",
            "action": "store_false",
            "dest": "posterior_features",
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
