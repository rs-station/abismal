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
)
