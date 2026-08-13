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
            "help": "Optional pre-normalization function for feed forward layers. The default is 'l2'",
            "default": "l2",
            "type": str,
            "choices": FeedForward.norm_dict.keys(),
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
        ("--normalizer-gain",),
        {
            "help": "Gain applied in the numerator of the 'l2'/'rms' feed forward normalizers. The default is 0.1",
            "default": 0.1,
            "type": float,
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
            "help": "Epsilon used in the feed forward layers' pre-normalization (see --normalizer). "
            "Defaults to None, in which case it falls back to --epsilon.",
            "default": None,
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
