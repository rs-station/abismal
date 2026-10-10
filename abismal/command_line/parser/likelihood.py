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
            "help": "Inflate the reported uncertainties, sigma'^2 = SdFac^2 (sigma^2 + SdB mu + (SdAdd mu)^2). "
                    "'ev11' learns SdFac, SdB and SdAdd with the model. 'mm24' fits SdFac and SdAdd "
                    "before training and after each epoch to pairwise differences of scaled, "
                    "equivalent observations. The default, 'none', uses the reported uncertainties.",
            "default": "none",
            "choices": ["none", "ev11", "mm24"],
        }
    ),
)
