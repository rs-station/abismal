title = "IO"
description = "Arguments controlling file inputs and outputs."


args_and_kwargs=(
    (
        (
            "-o",
            "--out-dir",
        ),{
        "help": "The directory in which to output results. The current working directory by default",
        "default": ".",
        "type": str,
        }
    ),

    (
        (
            "--num-cpus",
        ),{
        "help": "Number of CPUs to use for parsing CrystFEL .stream files with default 1.",
        "default": 1,
        "type": int,
        }
    ),

    (
        (
            "inputs",
        ),{
            "nargs": '+',
            "help": 'Either .stream files from CrystFEL or .refl and .expt files from dials',
        }
    ),

    (
        (
            "--wavelength",
        ),{
            "type": float,
            "default" : None,
            "help": 'Override the wavelengths inferred from the inputs.',
        }
    ),

    (
        (
            "--separate",
        ),{
        "help": "Merge the contents of each input file into a separate output file. ",
        "action": "store_true",
        }
    ),
)
