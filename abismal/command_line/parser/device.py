title = "Device options"
description = "Options dictating which hardware PyTorch uses and how training is run"

from argparse import _StoreTrueAction

class ListDevicesAction(_StoreTrueAction):
    def __call__(self, *args, **kwargs):
        from abismal.device import list_devices
        print("#############################################")
        print("# PyTorch can access the following devices #")
        print("#############################################")
        for dev in list_devices():
            print(f" - {dev}")
        from sys import exit
        exit()

args_and_kwargs = (
    (
        (
            "--gpu-id",
        ),{
            "help": "The numerical id of the GPU you wish to use. Ignored when no GPU is available.",
            "default": 0,
            "type": int,
        }
    ),
    (
        (
            "--device",
        ),{
            "help": "Run on this PyTorch device, for instance 'cpu', 'cuda:1' or 'mps'. "
                    "Overrides --gpu-id. By default abismal uses the GPU given by --gpu-id "
                    "when there is one and the CPU otherwise.",
            "default": None,
            "type": str,
        }
    ),
    (
        (
            "--compile",
            "--jit-compile",
        ),{
            "help": "Compile the training step with torch.compile: 'auto' (the default) "
                    "compiles on CUDA GPUs and not elsewhere, 'always' or 'never' override "
                    "that, and a bare --compile means 'always'. Compiling takes about half "
                    "a minute up front and then trains roughly a third faster in less GPU "
                    "memory.",
            "nargs": "?",
            "const": "always",
            "default": "auto",
            "choices": ["auto", "always", "never"],
            "dest": "compile",
        }
    ),
    (
        (
            "--precision",
        ),{
            "help": "Numerical precision of the scale model's networks. 'auto' (the "
                    "default) uses bfloat16 on CUDA GPUs with native bfloat16 support "
                    "(Ampere and newer) and float32 everywhere else. bfloat16 runs the "
                    "networks about twice as fast in less memory; everything outside "
                    "them stays float32.",
            "default": "auto",
            "choices": ["auto", "float32", "bfloat16"],
        }
    ),
    (
        (
            "--seed",
        ),{
            "help": "Seed PyTorch's random number generator, which draws the Monte Carlo "
                    "samples. Results still vary from run to run on a GPU, whose "
                    "reductions are not deterministic.",
            "default": None,
            "type": int,
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
            "--embed",
        ),{
            "help": "Embed into an IPython shell just before and just after the model is trained. Helpful for debugging.",
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
            "--verbosity",
            "--keras-verbosity",
        ),{
            "help": "0 is silent, 1 shows a progress bar and 2 prints one line per epoch. "
                    "The default is 1.",
            'type' : int,
            'default' : 1,
            "choices" : [0, 1 ,2],
            "dest" : "verbosity",
        }
    ),
)
