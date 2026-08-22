title = "Likelihood"
description = "Arguments affecting the likelihood term in the ELBO estimated by ABISMAL"

from abismal.command_line.parser.custom_types import float_or_none

args_and_kwargs=(
    (
        (
            "-t",
            "--studentt-dof",
        ),{
            "help": "Use a t-distributed error model with this many degrees of freedom. "
                    "The default is 32. Pass 'none' for a normal error model.",
            "type" : float_or_none,
            "default": 32.,
        }
    ),
)
