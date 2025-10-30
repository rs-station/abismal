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
            "--reference-mtz",
        ),{
            "type": str,
            "default" : None,
            "help": 'A reference mtz file which will be used to determine the reindexing operator.',
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

    (
        (
            "--separate-friedel-mates",
        ),{
        "help": "Separate the Friedel mates into ASU ID 0 (plus) and 1 (minus). "
                "This option is not compatible with --separate. ",
        "action": "store_true",
        }
    ),

    (
        (
            "--isigi-cutoff",
        ),{
            "type": float,
            "default" : None,
            "help": 'Discard reflections below this I/Sigma threshold.',
        }
    ),

    (
        (
            "--fractional-cell-tolerance",
        ),{
            "type": float,
            "default" : None,
            "help": 'Discard images with cells that deviate from the average or supplied cell by more than this fractional tolerance. ',
        }
    ),

)
