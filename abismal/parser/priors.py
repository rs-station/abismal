title = "Priors"
description = "Arguments governing the prior distributions"

args_and_kwargs = (
    (
        (
            "--kl-weight",
        ),{
            "help": "The strength of the structure factor prior distribution with default 0.1.",
            "default": 0.1,
            "type": float,
        }
    ),

    (
        (
            "--scale-kl-weight",
        ),{
            "help": "The strength of the scale prior distribution with default 1.0.",
            "default": 1.0,
            "type": float,
        }
    ),
)
