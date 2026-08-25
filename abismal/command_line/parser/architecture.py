title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

from abismal.layers import FeedForward

# Keras has no registry of its named activations to enumerate. `dir` over
# `tf_keras.activations` comes close, but it misses `leaky_relu` -- which resolves
# through `tfk.activations.get` and works fine -- so it would silently drop a
# usable option. This list is curated instead, and any of these may be given to
# either of the two transform slots. `FeedForward.get_transform` still accepts any
# keras activation; the restriction is the CLI's, so the choices render as a
# dropdown in the notebook GUI, which builds itself from `action.choices`.
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
