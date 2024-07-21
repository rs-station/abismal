title = "Ray options"
description = "Ray is used to parallelize CrystFEL .stream file loading."

args_and_kwargs = (
    (
        (
            "--ray-log-level",
        ),{
            "help": "Ray log level which defaults to 'ERROR'.",
            "default": 'ERROR',
            "type": str,
        }
    ),
)
