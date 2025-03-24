title = "Architecture"
description = "Arguments affecting the model architecture and dimensions"

def int_or_none_type(x):
    if x.lower() == 'none':
        return None
    x = int(x)
    if x <= 0:
        return None
    return x

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

    (
        (
            "--standardization-count-max",
        ),{
            "help": "Abismal uses Welford's algorithm to calculate the standard deviation of intensities. "
                    "After one pass over the data, updating the standard deviation may introduce noise. "
                    "So, typically the value is frozen after a number of steps specified by this parameter. "
                    "The default is 2000 steps, and 0 or None is interpreted as never freezing.",
            "default": 2000,
            "type": int_or_none_type,
        }
    ),
)
