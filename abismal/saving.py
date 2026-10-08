"""
Saving and loading abismal models.

A saved model is one `torch.save`d dict holding the model's config as JSON, its
state_dict, and optionally the optimizer's class, hyperparameters and state.
It contains only plain types and tensors, so it loads under torch's default
`weights_only=True` without trusting the file to run code.
"""
import json
import warnings
import torch
from abismal import serialization

FORMAT = "abismal-torch"
FORMAT_VERSION = 1


def _optimizer_config(optimizer):
    lr = optimizer.learning_rate
    return {
        "class_name": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "config": {
            # A schedule is a function; record the rate it had reached.
            "learning_rate": lr if not callable(lr) else optimizer.current_learning_rate(),
            "beta_1": optimizer.beta_1,
            "beta_2": optimizer.beta_2,
            "epsilon": optimizer.epsilon,
            "weight_decay": optimizer.weight_decay,
            "clipnorm": optimizer.clipnorm,
            "clipvalue": optimizer.clipvalue,
            "global_clipnorm": optimizer.global_clipnorm,
        },
    }


def save_model(model, path, optimizer=None):
    """
    Parameters
    ----------
    model : VariationalMergingModel
    path : str
    optimizer : AbismalOptimizer (optional)
        Saved with its state, so that training can resume where it stopped.
    """
    from abismal import __version__
    payload = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "abismal_version": __version__,
        "config": json.dumps(serialization.serialize(model)),
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_config"] = json.dumps(_optimizer_config(optimizer))
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def _is_keras(path):
    return str(path).endswith(".keras")


def _looks_like_keras(path):
    """A Keras v3 zip, whatever its name, which torch.load would choke on."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return "model.weights.h5" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


_CONVERT_HINT = (
    "Models saved by abismal 1.x (.keras) can be converted with the "
    "`abismal.convert` command or `abismal.convert.convert_keras`."
)


def _load(path, map_location):
    if _looks_like_keras(path):
        raise ValueError(f"{path} is an abismal 1.x (.keras) model. " + _CONVERT_HINT)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not an abismal model file. " + _CONVERT_HINT)
    return payload


def load_model(path, map_location="cpu"):
    """
    Rebuild a model saved with `save_model`, weights included.

    A `.keras` file from abismal 1.x is converted on the fly (see
    `abismal.convert`), with a warning; converting it once with the
    `abismal.convert` command avoids repeating that work.
    """
    if _is_keras(path):
        from abismal.convert import load_keras
        warnings.warn(
            f"{path} is an abismal 1.x model; converting it to the abismal 2 "
            "format in memory. Run `abismal.convert` on it to convert it once "
            "and for all.",
            stacklevel=2,
        )
        model, _ = load_keras(str(path))
        return model.to(map_location)
    payload = _load(path, map_location)
    model = serialization.deserialize(json.loads(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model


def load_optimizer(path, model, map_location="cpu", state=True):
    """
    Rebuild the optimizer saved alongside a model, attached to `model`'s
    trainable parameters, or return None if none was saved. With
    `state=False` only its class and hyperparameters are restored, and its
    moments start from zero.
    """
    if _is_keras(path):
        # The class and hyperparameters of the 1.x optimizer; its moments are
        # never converted, so `state` makes no difference.
        from abismal.convert import read_keras, _Weights, _optimizer
        config, weights = read_keras(str(path))
        return _optimizer(config, _Weights(weights), model)
    payload = _load(path, map_location)
    if "optimizer_config" not in payload:
        return None
    import importlib
    config = json.loads(payload["optimizer_config"])
    module, _, name = config["class_name"].rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    from abismal.training import parameter_groups
    optimizer = cls(parameter_groups(model), **config["config"])
    if state:
        optimizer.load_state_dict(payload["optimizer"])
    return optimizer
