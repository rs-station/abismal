title = "Likelihood"
description = "Arguments affecting the likelihood term in the ELBO estimated by ABISMAL"

args_and_kwargs=(
    (
        (
            "-t",
            "--studentt-dof",
        ),{
            "help": "Use a t-distributed error model with this many degrees of freedom.",
            "type" : float,
            "default": None,
        }
    ),
)
