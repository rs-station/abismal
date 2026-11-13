title = "Optimizer"
description = "Arguments affecting the optimization algorithm"

from abismal.optimizers.optimizer_dict import optimizer_dict

args_and_kwargs=(

    (
        (
            "--learning-rate",
        ),{
            "help": "Learning rate for Adam with default 1e-3.",
            "default": 1e-3,
            "type": float,
        }
    ),

    (
        (
            "--burnin",
        ),{
            "help": "The number of steps after which the step schedule drops the learning rate to "
                    "--learning-rate-final. The default drops it two thirds of the way through training "
                    "(epochs x steps per epoch), at step 20000 with the default 30 epochs of 1000 steps.",
            "default": None,
            "type": int,
        }
    ),

    (
        (
            "--learning-rate-final",
        ),{
            "help": "The learning rate after the drop (see --burnin and --lr-schedule), with default 1e-4. "
                    "Pass the same value as --learning-rate to keep the rate constant.",
            "default": 1e-4,
            "type": float,
        }
    ),

    (
        (
            "--lr-schedule",
        ),{
            "help": "How the learning rate goes from --learning-rate to --learning-rate-final (which this needs): "
                    "'step' (the default) drops it all at once after --burnin steps; 'cosine' lowers it along half "
                    "a cosine over all of training (epochs x steps per epoch) and ignores --burnin.",
            "default": "step",
            "choices": ("step", "cosine"),
        }
    ),

    (
        (
            "--beta-1",
        ),{
            "help": "First moment momentum parameter for Adam with default 0.9.",
            "default": 0.9,
            "type": float,
        }
    ),

    (
        (
            "--beta-2",
        ),{
            "help": "Second moment momentum parameter for Adam with default 0.999.",
            "default": 0.999,
            "type": float,
        }
    ),

    (
        (
            "--adam-epsilon",
        ),{
            "help": "A small constant for numerical stability with default 1e-9.",
            "default": 1e-7,
            "type": float
        }
    ),

    (
        (
            "--per-reflection-bias-correction",
        ),{
            "help": "Bias-correct each surrogate posterior parameter by its own number of (lazy) updates rather than "
                    "the global step count, so a rarely observed reflection takes Adam-sized steps from its first "
                    "observation instead of steps several times larger while its second moment warms up.",
            "action": "store_true",
            "default": False,
        }
    ),

    (
        (
            "--global-clipnorm",
        ),{
            "help": "Optionally apply gradient clipping with a global norm.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--clipnorm",
        ),{
            "help": "Optionally apply gradient clipping with a per-parameter norm.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--clip",
        ),{
            "help": "Optionally apply gradient clipping with a value.",
            "default": None,
            "type": float,
        }
    ),

    (
        (
            "--optimizer",
        ),{
            "help": "Choose the optimizer to use. The default is adam.",
            "default" : 'adam', 
            "type" : str.lower,
            "choices": optimizer_dict.keys(),
        }
    ),

    (
        (
            "--weight-decay",
        ),{
            "help": "Decoupled weight decay coefficient. Left unset, each optimizer keeps its "
                    "own default, which is 0.004 for adamw and none for the others; 0.0 recovers "
                    "plain Adam. Every optimizer here accepts it, but only adamw handles the lazy "
                    "variables correctly: it decays a structure factor once per update, whereas "
                    "the others decay every variable on every step, pulling rarely observed "
                    "reflections toward zero at a rate set by how seldom they appear in a batch "
                    "rather than by the objective. The flip side is that under adamw a reflection "
                    "seen one step in ten decays ten times more slowly than a dense weight at the "
                    "same coefficient, so the effective decay on merged structure factors varies "
                    "with multiplicity.",
            "default" : None,
            "type" : float,
        }
    ),
)
