title = "Training options"
description = "Options that deal with the specifics of model training"


args_and_kwargs = (
    (
        (
            "--mc-samples",
        ),{
            "help": "The number of monte carlo samples used to estimate gradients with default 256.",
            "default": 256,
            "type": int,
        }
    ),

    (
        (
            "--steps-per-epoch",
        ),{
            "help": "The number of gradient steps in an epoch dictating how often output can be saved. 1000 is the default.",
            "default": 1_000,
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
            "help": "The number of validation steps run at the close of each epoch with default 100.",
            "default": 100,
            "type": int,
        }
    ),

    (
        (
            "--sample-reflections-per-image",
        ),{
            "help": "The number of reflections which will be sampled in order to calculate the image representation with default 32.",
            "default": 32,
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
            "default": 100_000,
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
