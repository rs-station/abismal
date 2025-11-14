title = "Priors"
description = "Arguments governing the prior distributions"

from abismal.command_line.parser.custom_types import list_of_ints,list_of_ops,list_of_floats
from abismal.scaling.scaling import ImageScaler

args_and_kwargs = (
    (
        (
            "--kl-weight",
        ),{
            "help": "The strength of the structure factor prior distribution with default 1.0.",
            "default": 1.0,
            "type": float,
        }
    ),

    (
        (
            "--scale-kl-weight",
        ),{
            "help": "The strength of the scale prior distribution with default 1.0.",
            "default": 1.0,
            "type": float,
        }
    ),

    (
        (
            "--parents",
        ),{
            "help": "Set parent asu for the Multi-Wilson prior. This is used with --separate flag." 
                    "Supply parent asu ids as a comma separated list of integers. If an asu has no"
                    " parent, supply its own asu id. for example, --parents 0,0, indicates that "
                    "the first asu has no parent and the second asu is dependent on the first.",
            "default": None,
            "type": list_of_ints,
        }
    ),

    (
        (
            "-r",
            "--prior-correlation",
        ),{
            "help": "The prior correlation (r-value) for each ASU and its parent. Supply "
                    "comma-separated floating point Values. " 
                    "Values supplied for ASUs without a parent will be ignored. "
                    "Example: -r 0.0,0.99",
            "default": None,
            "type": list_of_floats,
        }
    ),

    (
        (
            "--reindexing-ops",
        ),{
            "help": 'Supply semicolon-separated reindexing ops which map from the child asu' 
                    'convention into the parent convention. ' 
                    'For example, --reindexing-ops "x,y,z;-x,-y,-z". ',
            "default": None,
            "type": list_of_ops,
        }
    ),

    (
        (
            "--prior-distribution",
        ),{
            "help": "The prior to use for structure factors or intensities.  "
                    "Wilson is the defalt",
            "default": 'Wilson',
            "type": str.lower,
            "choices" : ["wilson", "normal", 'halfnormal'],
        }
    ),

    (
        (
            "--scale-prior-distribution",
        ),{
            "help": "The scale prior to use.  "
                    "Cauchy is the defalt",
            "default": 'Cauchy',
            "type": str.lower,
            "choices" : ImageScaler.prior_dict.keys(),
        }
    ),
)
