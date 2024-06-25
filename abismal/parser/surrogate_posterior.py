title = "Surrogate posteriors"
description = "Arguments affecting the parameterization and initialization of surrogate posteriors"

args_and_kwargs = (
    (
        (
            "--init-scale",
        ),{
            "help": "The surrogate posteriors will be initialized with standard deviations equal to init_scale * mean(prior). The default is 0.01. ",
            "default": 0.01,
            "type": float,
        }
    ),

    (
        (
            "--anomalous",
        ),{
            "help": "Keep the two halves of reciprocal space separate during merging.",
            "action": 'store_true',
        }
    ),

    (
        (
            "--cell",
        ),{
            "help": "Use the specified unit cell.",
            "nargs": 6,
            "type": float,
            "default" : None,
        }
    ),

    (
        (
            "--space-group",
        ),{
            "help": "Use the specified space group instead of P1.",
            "type": str,
            "default" : None,
        }
    ),

    (
        (
            "-d",
            "--dmin",
        ),{
            "help": "The resolution at which to truncate the data",
            "required": True,
            "type": float,
        }
    ),
)
