title = "Likelihood"
description = "Arguments affecting the likelihood term in the ELBO estimated by ABISMAL"

from abismal.command_line.parser.custom_types import degrees_of_freedom

args_and_kwargs=(
    (
        (
            "-t",
            "--studentt-dof",
        ),{
            "help": "Degrees of freedom for the t-distributed error model, default 32. A t "
                    "distribution converges to a normal as this goes to infinity, so pass 0, "
                    "inf or none for a normal error model.",
            "type" : degrees_of_freedom,
            "default": 32.,
        }
    ),
)
