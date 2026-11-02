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
    (
        (
            "--error-model",
        ),{
            "help": "'ev11' learns the Evans (2011) error model Var(I) = a·σ² + b²·(s·Î)² with the rest of the model, "
                    "s·Î the predicted intensity (detached in the variance). The default, 'none', uses the reported "
                    "uncertainties.",
            "default": "none",
            "choices": ["none", "ev11"],
        }
    ),
    (
        (
            "--no-error-model-detach",
        ),{
            "help": "--error-model ev11: let the predicted intensity in the variance carry gradient to the scales and "
                    "the surrogate posterior (detached by default).",
            "dest": "error_model_detach",
            "action": "store_false",
        }
    ),
)
