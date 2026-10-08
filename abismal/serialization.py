"""
A minimal stand-in for Keras' object serialization.

Every abismal object that is saved with a model implements `get_config()`, which
returns plain JSON types plus nested `serialize()`d children, and is registered
here so `deserialize()` can find its class again from the saved name. Weights
are not part of the config; they travel separately in a torch `state_dict`.

Classes are keyed by their fully qualified name rather than their bare name:
the intensity and structure factor packages both define a `LogNormalPosterior`,
which Keras could only tell apart through hand-picked registration names.
"""

_REGISTRY = {}


def _key(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def register(cls):
    """Class decorator making `cls` available to `deserialize`."""
    _REGISTRY[_key(cls)] = cls
    return cls


def serialize(obj):
    """
    Return a JSON-compatible description of `obj`, from which `deserialize`
    rebuilds an equivalent object (without its weights).
    """
    if obj is None:
        return None
    cls = type(obj)
    key = _key(cls)
    if key not in _REGISTRY:
        raise TypeError(
            f"{key} is not registered with abismal.serialization.register, so a "
            "saved model containing it could not be loaded again."
        )
    return {"class_name": key, "config": obj.get_config()}


def deserialize(config):
    """Rebuild an object from the output of `serialize`."""
    if config is None:
        return None
    key = config["class_name"]
    try:
        cls = _REGISTRY[key]
    except KeyError:
        # The defining module may simply not have been imported yet. Every
        # registered class lives in an abismal module, so importing that module
        # is enough to register it.
        import importlib

        module, _, _ = key.rpartition(".")
        while module:
            try:
                importlib.import_module(module)
                break
            except ModuleNotFoundError:
                module, _, _ = module.rpartition(".")
        cls = _REGISTRY[key]
    return cls.from_config(dict(config["config"]))


class Serializable:
    """Mixin giving `from_config` the obvious default."""

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)
