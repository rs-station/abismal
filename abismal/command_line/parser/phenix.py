title = "phenix"
description = "Arguments for running PHENIX periodically during optimization."

from abismal.command_line.parser.custom_types import list_of_paths


args_and_kwargs=(
    (
        (
            "--eff-files",
        ),{
        "help": "Comma separated list of eff files.",
        "default": None,
        "type": list_of_paths,
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
