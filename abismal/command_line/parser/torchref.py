title = "torchref"
description = "Arguments for running torchref refinement periodically during optimization."


args_and_kwargs = (
    (
        (
            "--torchref-pdb",
        ), {
            "help": "Path to a starting PDB model. When set, torchref refines "
                    "this model against the merged structure factors every "
                    "--torchref-frequency epochs (a PyTorch-based alternative "
                    "to phenix.refine that runs on Colab).",
            "default": None,
            "type": str,
        }
    ),

    (
        (
            "--torchref-frequency",
        ), {
            "type": int,
            "default": 1,
            "help": "How often to run torchref in epochs (default=1).",
        }
    ),

    (
        (
            "--r-free-mtz",
        ), {
            "help": "MTZ supplying a fixed R-free set for torchref refinement. "
                    "abismal's merged output carries no R-free flags, so "
                    "without this torchref generates a fresh random set every "
                    "epoch and Rfree cannot be compared across epochs or "
                    "against phenix. Flags are matched on Miller index.",
            "default": None,
            "type": str,
        }
    ),

    (
        (
            "--r-free-value",
        ), {
            "help": "Integer flag value marking a FREE reflection in "
                    "--r-free-mtz, the equivalent of the phenix GUI's test "
                    "flag value. Inferred automatically when the flag column "
                    "holds only two values; required for multi-bin flag sets "
                    "(e.g. 0-19 with one bin nominated as the test set).",
            "default": None,
            "type": int,
        }
    ),

    (
        (
            "--torchref-wavelength",
        ), {
            "help": "Experimental wavelength in Angstroms, used by torchref for "
                    "the f'/f'' anomalous correction during refinement. Leave "
                    "unset to use torchref's default (1.0). Set to 0 to disable "
                    "anomalous refinement and force a Friedel-merged read.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--torchref-adp-mode",
        ), {
            "help": "ADP parametrization for torchref refinement. 'auto' "
                    "(default) refines anisotropically when the starting model "
                    "carries ANISOU records and isotropically otherwise.",
            "default": "auto",
            "choices": ["auto", "isotropic", "anisotropic"],
            "type": str,
        }
    ),

    (
        (
            "--torchref-adp-aniso-sigma",
        ), {
            "help": "Sigma on the deviatoric (anisotropy) channel of torchref's "
                    "SIMU restraint -- the dial that regularizes ADP tensor "
                    "shape, which is where all the extra anisotropic "
                    "parameters live. 'auto' (default) fits it once by "
                    "minimising Rfree, refit each epoch against that epoch's "
                    "own merged data. Ignored for isotropic runs.",
            "default": "auto",
            "type": str,
        }
    ),

    (
        (
            "--torchref-z-score",
        ), {
            "type": float,
            "default": 5.,
            "help": "Z-score cutoff for anomalous peak finding on torchref "
                    "output (default=5.0). Peak finding runs automatically "
                    "whenever the merged data are anomalous.",
        }
    ),
)
