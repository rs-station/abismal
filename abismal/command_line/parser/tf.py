title = "TensorFlow options"
description = "Options dictating how and what resources will be used by TF"

args_and_kwargs = (
    (
        (
            "--gpu-id",
        ),{
            "help": "The numerical id of the GPU you wish to use.",
            "default": 0,
            "type": int,
        }
    ),

    (
        (
            "--tf-log-level",
        ),{
            "help": "The tf log level is an int with default 3.",
            "default": 3,
            "type": int,
        }
    ),

    (
        (
            "--run-eagerly",
        ),{
            "help": "Run in eager mode.",
            "action" : "store_true",
        }
    ),

    (
        (
            "--jit-compile",
        ),{
            "help": "Compile with jit.",
            "action" : "store_true",
        }
    ),
)
