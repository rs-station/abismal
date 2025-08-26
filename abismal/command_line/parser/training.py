title = "Training options"
description = "Options that deal with the specifics of model training"


args_and_kwargs = (
    (
        (
            "--mc-samples",
        ),{
            "help": "The number of monte carlo samples used to estimate gradients with default 256.",
            "default": 32,
            "type": int,
        }
    ),

    (
        (
            "--steps-per-epoch",
        ),{
            "help": "Optional set the number of gradient steps in an epoch dictating how often output will be saved.",
            "default": None,
            "type": int,
        }
    ),

    (
        (
            "--epochs",
        ),{
            "help": "The number of training epochs to run with default 30.",
            "default": 30,
            "type": int,
        }
    ),

    (
        (
            "--validation-steps",
        ),{
            "help": "Optionally set the number of validation steps run at the close of each epoch.",
            "default": None,
            "type": int,
        }
    ),

    (
        (
            "--sample-reflections-per-image",
        ),{
            "help": "Optionally subsample the reflections going into the encoder. This can decrease memory usage."
                    "By default use all reflections.",
            "default": None,
            "type": int,
        }
    ),

    (
        (
            "--test-fraction",
        ),{
            "help": "The fraction of images reserved for validation with default 0.01.",
            "default": 0.01, "type": float,
        }
    ),

    (
        (
            "--shuffle-buffer-size",
        ),{
            "help": "The size of the shuffle buffer which randomizes the training data with default 100_000.",
            "default": 0,
            "type": int,
        }
    ),

    (
        (
            "--batch-size",
        ),{
            "help": "The size (number of images) in each training batch",
            "default": 100,
            "type": int,
        }
    ),

    (
        (
            "--disable-index-disambiguation",
        ),{
            "help": "Disable index disambiguation if applicable to the space group.",
            "action": "store_true",
        }
    ),


)
