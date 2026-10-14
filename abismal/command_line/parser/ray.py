title = "Ray options"
description = "Ray is used to parallelize CrystFEL .stream file loading."

args_and_kwargs = (
    (
        (
            "--ray-log-level",
        ),{
            "help": "Ignored. Kept so that older command lines and configs still parse (abismal no longer uses ray).",
            "default": 'ERROR',
            "type": str,
        }
    ),
)
