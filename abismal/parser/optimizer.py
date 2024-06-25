title = "Optimizer"
description = "Arguments affecting the optimization algorithm"


args_and_kwargs=(

    (
        (
            "--learning-rate",
        ),{
            "help": "Learning rate for Adam with default 1e-3.",
            "default": 1e-3,
            "type": float
        }
    ),

    (
        (
            "--beta-1",
        ),{
            "help": "First moment momentum parameter for Adam with default 0.9.",
            "default": 0.9,
            "type": float,
        }
    ),

    (
        (
            "--beta-2",
        ),{
            "help": "Second moment momentum parameter for Adam with default 0.999.",
            "default": 0.999,
            "type": float,
        }
    ),

    (
        (
            "--adam-epsilon",
        ),{
            "help": "A small constant for numerical stability with default 1e-9.",
            "default": 1e-9,
            "type": float
        }
    ),

    (
        (
            "--global-clipnorm",
        ),{
            "help": "Optionally apply gradient clipping with a global norm.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--clipnorm",
        ),{
            "help": "Optionally apply gradient clipping with a per-parameter norm.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--clip",
        ),{
            "help": "Optionally apply gradient clipping with a value.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--amsgrad",
        ),{
            "help": "Optionally use the amsgrad variant of Adam.",
            "action": 'store_true',
        }
    ),
)
