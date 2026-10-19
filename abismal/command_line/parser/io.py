title = "IO"
description = "Arguments controlling file inputs and outputs."

from pathlib import Path
from abismal.command_line.parser.custom_types import directory


from abismal.io.parallel import available_cpus

args_and_kwargs=(
    (
        (
            "-o",
            "--out-dir",
        ),{
        "help": "The directory in which to output results. The current working directory by default",
        "default": Path("."),
        "type": directory,
        }
    ),

    (
        (
            "--num-cpus",
        ),{
        "help": "How many processes may parse input files in parallel. The default is the number "
        "of CPUs this process may use (its CPU affinity, which srun --cpu-bind sets). Parsing is "
        "split by file for DIALS input and by groups of chunks for CrystFEL .stream files; 1 parses "
        "serially.",
        "default": available_cpus(),
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
            "type": Path,
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

    (
        (
            "--mtz-metadata",
        ),{
            "type": str,
            "default" : None,
            "help": 'Comma-separated list of metadata keys for mtz files. ',
        }
    ),
)
