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
            "--run-eagerly",
        ),{
            "help": "Run in eager mode.",
            "action" : "store_true",
        }
    ),
)
