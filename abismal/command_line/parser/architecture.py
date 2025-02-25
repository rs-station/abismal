title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

args_and_kwargs=(
    (
        (
            "--d-model",
        ),{
        "help": "The number of channels in the model with default 32.",
        "default" : 256,
        "type": int,
        }
    ),

    (
        (
            "--layers",
        ),{
            "help": "The number of feedfoward layers with default 20.",
            "default": 5,
            "type": int,
        }
    ),

    (
        (
            "--activation",
        ),{
            "help": "The name of the activation function used in the scale model. The default is 'leaky_relu'",
            "default": "leaky_relu",
            "type": str,
        }
    ),

    (
        (
            "--epsilon",
        ),{
            "help": "A small constant for numerical stability.",
            "default": 1e-12,
            "type": float,
        }
    ),
)
