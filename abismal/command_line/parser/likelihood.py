title = "Likelihood"
description = "Arguments affecting the likelihood term in the ELBO estimated by ABISMAL"

args_and_kwargs=(
    (
        (
            "-t",
            "--studentt-dof",
        ),{
            "help": "Degrees of freedom for the t-distributed likelihood.",
            "type" : float,
            "default": 32,
        }
    ),
    (
        (
            "-l",
            "--likelihood",
        ),{
            "help": "The likelihood function to use with normal being the default.",
            "type" : str.lower,
            "choices" : ['normal', 'studentt', 'adaptive_studentt', 'ev11', 'leastsquares'],
            "default": 'normal',
        }
    ),
)
