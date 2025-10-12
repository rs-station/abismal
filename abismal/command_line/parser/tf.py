title = "TensorFlow options"
description = "Options dictating how and what resources will be used by TF"

from argparse import _StoreTrueAction
class ListDevicesAction(_StoreTrueAction):
    def __call__(self, *args, **kwargs):
        import tensorflow as tf
        print("###############################################")
        print("# TensorFlow can access the following devices #")
        print("###############################################")
        for dev in tf.config.list_physical_devices():
            print(f" - {dev.device_type}: {dev.name}")
        from sys import exit
        exit()

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

    (
        (
            "--debug",
        ),{
            "help": "Set various behaviors helpful for debugging.",
            "action" : "store_true",
        }
    ),

    (
        (
            "--list-devices",
        ),{
            "help": "List accelerator devices and exit.",
            "action" : ListDevicesAction,
        }
    ),

    (
        (
            "--keras-verbosity",
        ),{
            "help": "Keras Model.fit verbose level. See the docs for more info: "
            "https://keras.io/2.18/api/models/model_training_apis/#fit-method",
            'type' : int,
            'default' : 1,
            "choices" : [0, 1 ,2],
        }
    ),
)

