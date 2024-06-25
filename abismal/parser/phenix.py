title = "phenix"
description = "Arguments for running PHENIX periodically during optimization."


args_and_kwargs=(
    (
        (
            "--eff-files",
        ),{
        "help": "Comma separated list of eff files.",
        "default": None,
        "type": str,
        }
    ),

    (
        (
            "--phenix-frequency",
        ),{
            "type": int,
            "default" : 1, 
            "help": 'How often to run phenix in epochs (default=1).',
        }
    ),
)
