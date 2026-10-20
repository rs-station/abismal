"""
Saving and loading abismal models.

A saved model is one `torch.save`d dict holding the command-line arguments the
model was built from, the few facts about the data its structure depends on
(see `abismal.build.facts_from`), its state_dict, and optionally the
optimizer's state. Loading rebuilds the model with `abismal.build.build_model`
and then loads the weights, so model classes need no serialization code. The
file contains only plain types and tensors, so it loads under torch's default
`weights_only=True` without trusting the file to run code.
"""
from argparse import Namespace
from pathlib import PurePath

import torch

FORMAT = "abismal-torch"
FORMAT_VERSION = 2


def _plain(value):
    """`value` as types torch.load(weights_only=True) accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def save_model(model, path, optimizer=None):
    """
    Parameters
    ----------
    model : VariationalMergingModel
        A model from `abismal.build.build_model`, which records what it was
        built from.
    path : str
    optimizer : AbismalOptimizer (optional)
        Saved with its state, so that training can resume where it stopped.
    """
    spec = getattr(model, "spec", None)
    if spec is None:
        raise ValueError(
            "This model was not built by abismal.build.build_model, so there is "
            "nothing to rebuild it from. Save its state_dict with torch.save instead."
        )
    from abismal import __version__
    payload = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "abismal_version": __version__,
        "args": _plain(spec["args"]),
        "facts": _plain(spec["facts"]),
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def _load(path, map_location):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not an abismal model file.")
    return payload


# Options added after checkpoints began recording their arguments, with the value
# that reproduces what a checkpoint saved before them was built with. Today's
# default would not: --hidden-multiplier now defaults to 2/3, but those models
# had hidden layers twice --d-model wide.
_BEFORE_THEY_EXISTED = {"hidden_multiplier": 2.0}


def _args(payload, path):
    """The saved arguments over today's defaults, so new options keep theirs."""
    if "args" not in payload:
        raise ValueError(
            f"{path} was saved by an earlier abismal that did not record its "
            "arguments, so the model cannot be rebuilt from it. Its weights can "
            "still initialize a run through --scale-init-file or "
            "--posterior-init-file."
        )
    from abismal.command_line.parser import parser
    args = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    args.update(_BEFORE_THEY_EXISTED)
    args.update(payload["args"])
    return Namespace(**args)


def load_model(path, map_location="cpu"):
    """
    Rebuild a model saved with `save_model`, weights included.
    """
    from abismal.build import build_model
    payload = _load(path, map_location)
    model = build_model(_args(payload, path), payload["facts"])
    model.load_state_dict(payload["state_dict"])
    return model.to(map_location)


def load_args(path):
    """The command-line arguments a saved model was trained with, as a Namespace."""
    return _args(_load(path, "cpu"), path)


def load_optimizer(path, model, map_location="cpu", state=True):
    """
    Rebuild the optimizer saved alongside a model, attached to `model`'s
    trainable parameters, or return None if none was saved. With
    `state=False` only its class and hyperparameters are restored, and its
    moments start from zero.
    """
    from abismal.build import build_optimizer
    payload = _load(path, map_location)
    if "optimizer" not in payload:
        return None
    optimizer = build_optimizer(_args(payload, path), model)
    if state:
        optimizer.load_state_dict(payload["optimizer"])
    return optimizer


def load_weights(path, module, prefix):
    """
    Load the weights saved under `prefix` (for instance "scale_model") into
    `module`. Works for checkpoints from any abismal version, since it uses
    only the state_dict.
    """
    state_dict = _load(path, "cpu")["state_dict"]
    head = prefix + "."
    module.load_state_dict({k[len(head):]: v for k, v in state_dict.items() if k.startswith(head)})
