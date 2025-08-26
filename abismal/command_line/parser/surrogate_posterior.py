title = "Surrogate posteriors"
description = "Arguments affecting the parameterization and initialization of surrogate posteriors"

from abismal.scaling.scaling import ImageScaler

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
            "--posterior-type",
        ),{
            "help": "What type of posterior parameterization to use. The "
                    "default is structure_factor.",
            "type": str.lower,
            "default": "structure_factor",
            "choices" : ['structure_factor', 'intensity'],
        }
    ),

    (
        (
            "--posterior-distribution",
        ),{
            "help": "Define the type of posterior distribution. "
                    "The default is foldednormal. ",
            "type": str.lower,
            "default" : "foldednormal",
            "choices" : ["normal", "foldednormal", "rice", "gamma", "truncatednormal"],
        }
    ),


    (
        (
            "--posterior-rank",
        ),{
            "help": "This parameter makes the normal posterior low-rank multivariate. "
                    "By default, this is 1 (univariate). ",
            "type": int,
            "default" : 1,
        }
    ),

    (
        (
            "--scale-posterior-distribution",
        ),{
            "help": "Define the type of posterior distribution for scales. "
                    "The default is normal. ",
            "type": str.lower,
            "default" : "normal",
            "choices" : ImageScaler.posterior_dict.keys(),
        }
    ),

    (
        (
            "--scale-posterior-bijector",
        ),{
            "help": "Define the bijector used to enforce positivity of surrogate parameters. ",
            "type": str.lower,
            "default" : "softplus",
            "choices" : ImageScaler.bijector_dict.keys(),
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

    (
        (
            "--freeze-scales",
        ),{
            "help": "Don't optimize the scaling model.",
            "action": 'store_true',
        }
    ),

    (
        (
            "--freeze-posterior",
        ),{
            "help": "Don't optimize the structure factors / intensities.",
            "action": 'store_true',
        }
    ),

    (
        (
            "--posterior-init-file",
        ),{
            "help": "A `.keras` model file from which to initialize the intensity or structure factors weights.",
            "type": str,
            "default": None,
        }
    ),

    (
        (
            "--scale-init-file",
        ),{
            "help": "A `.keras` model file from which to initialize the scale model weights.",
            "type": str,
            "default": None,
        }
    ),
)
