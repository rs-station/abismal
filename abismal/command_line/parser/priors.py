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
            "--kl-reflection-weights",
        ),{
            "help": "Weight each observation's structure factor KL by its reflection's inverse observation count, "
                    "1/n_h, each image counted once. 'exact': 1/n_h, the ELBO itself, a reflection's prior counting "
                    "once against its n_h likelihood terms (default). 'standardized': 1/n_h over its mean over the reflections, "
                    "--kl-weight setting the overall strength. 'none': no weight, the prior counting once "
                    "per observation. Independent posteriors only; others ignore it.",
            "default": "exact",
            "choices": ["none", "standardized", "exact"],
        }
    ),

    (
        (
            "--posterior-trend",
        ),{
            "help": "Learn a falloff T(d) that multiplies the surrogate posterior's intensities in the likelihood and "
                    "the output, the posterior and its KL staying on a flat scale. 'wilson': log T = log k - b/d^2. "
                    "'spline' (default): log T piecewise linear in 1/d^2. 'none': no trend.",
            "default": "spline",
            "choices": ["none", "wilson", "spline"],
        }
    ),

    (
        (
            "--posterior-trend-knots",
        ),{
            "help": "--posterior-trend spline: the number of knots, default 16.",
            "default": 16,
            "type": int,
        }
    ),

    (
        (
            "--posterior-trend-anchor",
        ),{
            "help": "--posterior-trend: 'lowres' holds T at 1 at the lowest resolution (the spline's first knot, or k = 1 "
                    "for 'wilson'), so the trend carries the falloff's shape and the scales its level. 'none' (default): "
                    "the level is free, and can drift against the scales when nothing else pins them.",
            "default": "none",
            "choices": ["none", "lowres"],
        }
    ),

    (
        (
            "--posterior-trend-aniso",
        ),{
            "help": "--posterior-trend spline: add an anisotropic temperature factor, log T -= s^T B s / 2, with B traceless "
                    "and restricted to what the Laue symmetry allows. 'global': one B. 'bins': one B per knot, "
                    "interpolated in 1/d^2. 'none' (default): isotropic.",
            "default": "none",
            "choices": ["none", "global", "bins"],
        }
    ),

    (
        (
            "--scale-kl-weight",
        ),{
            "help": "The strength of the scale prior distribution with default 0.1.",
            "default": 0.1,
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
            "choices" : ["wilson", "normal", 'halfnormal', 'empirical_wilson', 'auto_wilson'],
        }
    ),

    (
        (
            "--auto-wilson-fixed-scale",
        ),{
            "help": "With --prior-distribution auto_wilson, learn only the B factor of "
                    "the Wilson prior and keep its scale k at 1.",
            "action": "store_true",
        }
    ),

    (
        (
            "--scale-prior-distribution",
        ),{
            "help": "The scale prior to use.  "
                    "lognormal is the defalt",
            "default": 'lognormal',
            "type": str.lower,
            "choices" : ImageScaler.prior_dict.keys(),
        }
    ),

    (
        (
            "--scale-prior-mode",
        ),{
            "help": "'fixed' (the default): every observation's scale prior is --scale-prior-distribution. "
                    "'decoded': each observation's prior is LogNormal(mu, sigma^2) with mu and sigma decoded from "
                    "the leave-one-out pooled image representation (the rest of its image), under a LogNormal(0, 1) "
                    "hyperprior counted once per observation. It ignores --scale-prior-distribution.",
            "default": 'fixed',
            "choices": ['fixed', 'decoded'],
        }
    ),

    (
        (
            "--no-scale-prior-detach",
        ),{
            "help": "With --scale-prior-mode decoded, let the scale KL's gradient reach the shared image "
                    "representation through the prior decoder. By default it is stopped there.",
            "action": "store_false",
            "dest": "scale_prior_detach",
        }
    ),
)
