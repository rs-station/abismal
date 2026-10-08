"""
Convert models saved by abismal 1.x (TensorFlow, `.keras`) to abismal 2 (`.pt`).

A `.keras` file is a zip holding `config.json`, the nested Keras config of the
VariationalMergingModel, and `model.weights.h5`, every variable stored by
position under the path of the layer that owns it. Both are read here without
TensorFlow: the config is mapped class by class onto the PyTorch constructors,
and each HDF5 dataset onto the state_dict entry it corresponds to.

Nothing is dropped quietly. A class, config key or weight the converter does not
know raises `ConversionError`, as does a state_dict entry that nothing in the
file filled (unless `allow_missing=True`).

The optimizer's class and hyperparameters carry over, but not its moments or
step count: a converted model resumes training from fresh Adam moments, as
`abismal.cchalf` does anyway.

Reading HDF5 needs h5py, which is imported only here. Install it with
`pip install "abismal[convert]"` (or just `pip install h5py`).

Command line:

    abismal.convert epoch_30.keras              # writes epoch_30.pt
    abismal.convert old.keras -o new.pt
"""
import io
import json
import re
import zipfile
import warnings
import numpy as np


class ConversionError(ValueError):
    """The .keras file holds something this converter cannot map faithfully."""


# Keys tf_keras adds to every layer's config; they mean nothing to abismal 2.
_KERAS_LAYER_KEYS = ("name", "trainable", "dtype", "build_config")


def _require_h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "Converting an abismal 1.x .keras model needs h5py, which abismal "
            "does not install by default. Install it with "
            "`pip install \"abismal[convert]\"` or `pip install h5py`."
        ) from e
    return h5py


def read_keras(path):
    """
    Read a `.keras` file without TensorFlow.

    Returns
    -------
    config : dict
        The parsed `config.json`.
    weights : dict
        Every dataset in `model.weights.h5`, by its HDF5 path, as numpy arrays.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            missing = {"config.json", "model.weights.h5"} - names
            if missing:
                raise ConversionError(
                    f"{path} is a zip but not a Keras v3 model file: it has no "
                    f"{', '.join(sorted(missing))}."
                )
            config = json.loads(z.read("config.json"))
            h5bytes = z.read("model.weights.h5")
    except zipfile.BadZipFile as e:
        raise ConversionError(
            f"{path} is not a .keras file (not a zip archive). Models saved in "
            "the older TensorFlow SavedModel or .h5 formats are not supported; "
            "re-save them as .keras with abismal 1.x first."
        ) from e

    h5py = _require_h5py()
    weights = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            weights[name] = np.asarray(obj[()])

    with h5py.File(io.BytesIO(h5bytes), "r") as f:
        f.visititems(visit)
    return config, weights


def _class_key(cfg):
    """`module.ClassName` of a serialized Keras object, as abismal 1.x wrote it."""
    if not isinstance(cfg, dict) or "class_name" not in cfg:
        raise ConversionError(f"Expected a serialized Keras object, got {cfg!r:.200}")
    module = cfg.get("module")
    if module is None:
        raise ConversionError(
            f"The config of {cfg['class_name']} has no 'module'. This file was "
            "probably written by a Keras version older than abismal 1.x used, "
            "and cannot be converted."
        )
    return f"{module}.{cfg['class_name']}"


def _layer_config(cfg, allowed, what):
    """The config of `cfg` without Keras' bookkeeping keys, checked against `allowed`."""
    config = {k: v for k, v in cfg["config"].items() if k not in _KERAS_LAYER_KEYS}
    unknown = set(config) - set(allowed)
    if unknown:
        raise ConversionError(
            f"{what} has config keys the converter does not know how to map: "
            f"{sorted(unknown)}."
        )
    return config


def _tensor(value):
    """A number, or a Keras-serialized tensor/array, as float32 numpy."""
    if isinstance(value, dict):
        if value.get("class_name") in ("__tensor__", "__numpy__"):
            return np.asarray(value["config"]["value"], dtype="float32")
        raise ConversionError(f"Cannot interpret {value!r:.200} as a tensor.")
    return np.asarray(value, dtype="float32")


class _Weights:
    """The HDF5 datasets, recording which ones have been mapped."""
    def __init__(self, arrays):
        self.arrays = arrays
        self.used = set()

    def __contains__(self, path):
        return path in self.arrays

    def get(self, path):
        if path not in self.arrays:
            raise ConversionError(f"model.weights.h5 has no {path!r}.")
        self.used.add(path)
        return self.arrays[path]

    def own_vars(self, prefix):
        """The paths of `prefix`'s own variables, `prefix/vars/0`, `/1`, ... in order."""
        pattern = re.compile(re.escape(prefix) + r"/vars/(\d+)$")
        found = {}
        for path in self.arrays:
            m = pattern.match(path)
            if m:
                found[int(m.group(1))] = path
        return [found[i] for i in sorted(found)]

    def has_group(self, prefix):
        return any(p.startswith(prefix + "/") for p in self.arrays)

    def unused(self):
        """Datasets nothing was mapped from, ignoring optimizer state and metrics."""
        out = []
        for path in self.arrays:
            if path in self.used:
                continue
            parts = path.split("/")
            if parts[0] == "optimizer" or "metrics" in parts:
                continue
            out.append(path)
        return sorted(out)


# ---------------------------------------------------------------------------
# Config -> torch objects
# ---------------------------------------------------------------------------

_RAC = "abismal.symmetry.reciprocal_asu"

# Every class kept its module path and name, so a 1.x `module.ClassName` is
# also its abismal 2 registry key. Posteriors map to the state_dict names of
# their variables in the order Keras stored them: trainable first, in creation
# order, then `seen`, the only non-trainable one. TFP's TransformedVariable
# stored its *pretransformed* value, which for Chain([Shift(s), Exp()]) is
# log(value - s) -- exactly what PositiveParameter.raw holds.
_POSITIVE = ["loc.raw", "scale.raw"]
_POSTERIORS = {
    "abismal.surrogate_posterior.structure_factor.folded_normal.FoldedNormalPosterior": _POSITIVE,
    "abismal.surrogate_posterior.structure_factor.rice.RicePosterior": _POSITIVE,
    "abismal.surrogate_posterior.structure_factor.truncated_normal.TruncatedNormalPosterior": _POSITIVE,
    "abismal.surrogate_posterior.structure_factor.normal.NormalPosterior": _POSITIVE,
    "abismal.surrogate_posterior.structure_factor.normal.MultivariateNormalPosterior": ["loc.raw", "_scale.raw", "scale_update"],
    "abismal.surrogate_posterior.structure_factor.nakagami.NakagamiPosterior": ["spread.raw", "concentration.raw"],
    "abismal.surrogate_posterior.structure_factor.lognormal.LogNormalPosterior": ["loc", "scale.raw"],
    "abismal.surrogate_posterior.intensity.folded_normal.FoldedNormalPosterior": _POSITIVE,
    "abismal.surrogate_posterior.intensity.normal.NormalPosterior": _POSITIVE,
    "abismal.surrogate_posterior.intensity.normal.MultivariateNormalPosterior": ["loc.raw", "_scale.raw", "scale_update"],
    "abismal.surrogate_posterior.intensity.gamma.GammaPosterior": ["rate.raw", "concentration.raw"],
    "abismal.surrogate_posterior.intensity.lognormal.LogNormalPosterior": ["loc", "scale.raw"],
}
_POSTERIOR_KEYS = ("rac", "epsilon", "independent", "rank", "concentration_min")

# Priors: the config keys passed on, and their variables in stored order.
_PRIORS = {
    "abismal.prior.structure_factor.wilson.WilsonPrior": (("sigma",), []),
    "abismal.prior.intensity.wilson.WilsonPrior": (("sigma",), []),
    # AutoWilson's config carries the Σ it had *computed* from b and k; it is
    # recomputed from those, so the stored value is not used.
    "abismal.prior.structure_factor.wilson.AutoWilsonPrior": ((), ["b", "_k"]),
    "abismal.prior.intensity.wilson.AutoWilsonPrior": ((), ["b", "_k"]),
    "abismal.prior.structure_factor.wilson.MultiWilsonPrior": (("correlation", "sigma"), []),
    "abismal.prior.normal.NormalPrior": (("loc", "scale"), []),
    "abismal.prior.normal.MultivariateNormalPrior": (("loc", "scale"), []),
    "abismal.prior.normal.HalfNormalPrior": (("scale",), []),
}
# Buffers the prior constructors fill from the config rather than the weights.
_PRIOR_BUFFERS_FROM_CONFIG = ("prior._sigma", "prior.sigma", "prior._correlation")

_LIKELIHOODS = {
    "abismal.likelihood.student.StudentTLikelihood": ("degrees_of_freedom",),
    "abismal.likelihood.normal.NormalLikelihood": (),
    "abismal.likelihood.normal.LeastSquaresLikelihood": (),
    "abismal.likelihood.normal.WeightedLeastSquaresLikelihood": (),
    "abismal.likelihood.laplace.LaplaceLikelihood": (),
}

_OPTIMIZERS = {
    "abismal.optimizers.adam.Adam": "Adam",
    "abismal.optimizers.adamw.AdamW": "AdamW",
    "abismal.optimizers.adabelief.AdaBelief": "AdaBelief",
}


def _registered(key):
    """The abismal 2 class registered under `key` (same module path as 1.x)."""
    from abismal import serialization
    module, _, _ = key.rpartition(".")
    import importlib
    importlib.import_module(module)
    try:
        return serialization._REGISTRY[key]
    except KeyError:
        raise ConversionError(f"abismal 2 has no class {key}.") from None


def _build_rac(cfg, cache):
    """A ReciprocalASUCollection/Graph from its 1.x config, shared between users."""
    marker = json.dumps(cfg, sort_keys=True, default=str)
    # Keras tags repeated objects with a shared_object_id that differs between
    # copies; it is not part of what the object is.
    marker = re.sub(r'"shared_object_id": \d+', "", marker)
    if marker in cache:
        return cache[marker]

    from abismal.symmetry.reciprocal_asu import (
        ReciprocalASU, ReciprocalASUCollection, ReciprocalASUGraph,
    )
    key = _class_key(cfg)
    config = {k: v for k, v in cfg["config"].items() if k not in _KERAS_LAYER_KEYS}
    rasu_keys = sorted(
        (k for k in config if re.fullmatch(r"rasu_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    if not rasu_keys:
        raise ConversionError(f"{key} config lists no reciprocal ASUs.")
    rasus = []
    for k in rasu_keys:
        rcfg = config.pop(k)
        if _class_key(rcfg) != f"{_RAC}.ReciprocalASU":
            raise ConversionError(f"Unexpected {_class_key(rcfg)} in {key}.")
        r = _layer_config(rcfg, ("cell", "spacegroup", "dmin", "anomalous"), "ReciprocalASU")
        rasus.append(ReciprocalASU(r["cell"], r["spacegroup"], r["dmin"], r.get("anomalous", True)))

    if key == f"{_RAC}.ReciprocalASUGraph":
        unknown = set(config) - {"parents", "reindexing_ops"}
        if unknown:
            raise ConversionError(f"ReciprocalASUGraph has unknown config keys {sorted(unknown)}.")
        rac = ReciprocalASUGraph(*rasus, **config)
    elif key == f"{_RAC}.ReciprocalASUCollection":
        if config:
            raise ConversionError(f"ReciprocalASUCollection has unknown config keys {sorted(config)}.")
        rac = ReciprocalASUCollection(*rasus)
    else:
        raise ConversionError(f"Cannot convert reciprocal ASU collection {key}.")
    cache[marker] = rac
    return rac


def _build_posterior(cfg, racs):
    key = _class_key(cfg)
    if key not in _POSTERIORS:
        raise ConversionError(f"Cannot convert surrogate posterior {key}.")
    config = _layer_config(cfg, _POSTERIOR_KEYS, key)
    config["rac"] = _build_rac(config["rac"], racs)
    return _registered(key)(**config), _POSTERIORS[key] + ["seen"]


def _build_prior(cfg, racs):
    key = _class_key(cfg)
    if key not in _PRIORS:
        raise ConversionError(f"Cannot convert prior {key}.")
    keep, names = _PRIORS[key]
    config = _layer_config(cfg, ("rac", "sigma") + keep, key)
    kwargs = {"rac": _build_rac(config["rac"], racs)}
    for k in keep:
        if k not in config:
            continue
        v = config[k]
        if k == "sigma":
            v = _tensor(v)
            if v.ndim > 0 and v.shape[0] != kwargs["rac"].asu_size:
                raise ConversionError(
                    f"{key} has a Σ of length {v.shape[0]}, but its ASU has "
                    f"{kwargs['rac'].asu_size} reflections."
                )
        kwargs[k] = v
    return _registered(key)(**kwargs), names


def _build_likelihood(cfg):
    key = _class_key(cfg)
    if key not in _LIKELIHOODS:
        raise ConversionError(f"Cannot convert likelihood {key}.")
    config = _layer_config(cfg, _LIKELIHOODS[key], key)
    return _registered(key)(**config)


def _scale_model_keys():
    import inspect
    from abismal.scaling.scaling import ImageScaler
    params = inspect.signature(ImageScaler.__init__).parameters
    return tuple(k for k in params if k not in ("self", "metadata_length"))


def _metadata_length(model_cfg, weights):
    """The number of metadata columns, which 1.x inferred at build time."""
    path = "scale_model/layers/dense_1/vars/0"  # input_scale's kernel
    if path in weights:
        return int(weights.arrays[path].shape[0])
    sm = model_cfg["config"].get("standardize_metadata")
    if sm and "build_config" in sm:
        return int(sm["build_config"]["input_shape"][-1])
    if "build_config" in model_cfg:
        return int(model_cfg["build_config"]["input_shape"][4][-1])
    if "standardize_metadata/vars/0" in weights:
        return int(weights.arrays["standardize_metadata/vars/0"].shape[-1])
    return None


# ---------------------------------------------------------------------------
# Weights -> state_dict
# ---------------------------------------------------------------------------

def _feed_forward_paths(prefix, depth, gated):
    """Keras' paths for the Dense kernels of each residual block of a Sequential."""
    base = "glu_feed_forward" if gated else "feed_forward"
    names = ["ff1", "ff2", "ff3"] if gated else ["ff1", "ff2"]
    out = []
    for i in range(depth):
        layer = base if i == 0 else f"{base}_{i}"
        for name in names:
            out.append((f"{prefix}/layers/{layer}/{name}/vars/0", f"{i}.{name}.weight"))
    return out


def _scale_model_map(scale_model):
    """(h5 path, state_dict keys, transpose) for every ImageScaler weight."""
    sm = "scale_model"
    entries = [
        # Unbuilt containers are named by class, in creation order, skipping
        # those already saved under an attribute (output_prior, output_dense,
        # scale_network): so input_image is always `dense`, input_scale `dense_1`.
        (f"{sm}/layers/dense/vars/0", ["scale_model.input_image.weight"], True),
        (f"{sm}/layers/dense_1/vars/0", ["scale_model.input_scale.weight"], True),
        (f"{sm}/output_dense/vars/0", ["scale_model.output_dense.weight"], True),
    ]
    if scale_model.output_bias:
        entries.append((f"{sm}/output_dense/vars/1", ["scale_model.output_dense.bias"], False))
    if scale_model.optimize_prior_scale:
        entries.append((f"{sm}/output_prior/vars/0", ["scale_model.output_prior.weight"], True))
        if scale_model.output_bias:
            entries.append((f"{sm}/output_prior/vars/1", ["scale_model.output_prior.bias"], False))

    # Keras walks attributes in reverse alphabetical order, so the network
    # saved under `scale_network` is the one both attributes name when they
    # are shared; unshared, the image network is reached through the layer
    # list and lands under `layers/sequential`.
    depth, gated = scale_model.mlp_depth, scale_model.gated
    for path, key in _feed_forward_paths(f"{sm}/scale_network", depth, gated):
        keys = [f"scale_model.scale_network.{key}"]
        if scale_model.share_weights:
            keys.append(f"scale_model.image_network.{key}")
        entries.append((path, keys, True))
    if not scale_model.share_weights:
        for path, key in _feed_forward_paths(f"{sm}/layers/sequential", depth, gated):
            entries.append((path, [f"scale_model.image_network.{key}"], True))
    return entries


def _assign(state, key, value, source):
    import torch
    if key not in state:
        raise ConversionError(f"The converted model has no state_dict entry {key!r} (for {source}).")
    target = state[key]
    value = torch.as_tensor(np.asarray(value)).to(target.dtype)
    if tuple(value.shape) != tuple(target.shape):
        raise ConversionError(
            f"{source} has shape {tuple(value.shape)}, but {key} is "
            f"{tuple(target.shape)}."
        )
    state[key] = value


def _fill_state(model, weights, posterior_names, prior_names, allow_missing):
    state = model.state_dict()
    filled = set()

    def put(keys, value, source):
        for key in keys:
            _assign(state, key, value, source)
            filled.add(key)

    for path, keys, transpose in _scale_model_map(model.scale_model):
        if path not in weights:
            continue  # reported below as an unfilled state_dict entry
        value = weights.get(path)
        put(keys, value.T if transpose else value, path)

    for group, module, names in [
        ("surrogate_posterior", "surrogate_posterior", posterior_names),
        ("prior", "prior", prior_names),
    ]:
        paths = weights.own_vars(group)
        if not paths and not names:
            continue
        if not paths and allow_missing:
            continue
        if len(paths) != len(names):
            raise ConversionError(
                f"{group} stores {len(paths)} variables ({', '.join(paths) or 'none'}), "
                f"but {type(getattr(model, module)).__name__} expects {len(names)} "
                f"({', '.join(names)})."
            )
        for path, name in zip(paths, names):
            put([f"{module}.{name}"], weights.get(path), path)

    for group in ("standardize_intensity", "standardize_metadata"):
        if getattr(model, group) is None:
            if weights.has_group(group) and weights.own_vars(group):
                raise ConversionError(f"{group} has weights but the model does not standardize.")
            continue
        paths = weights.own_vars(group)
        if not paths:
            continue
        if len(paths) != 4:
            raise ConversionError(f"{group} stores {len(paths)} variables, expected 4 (mean, m2, count, frozen).")
        for path, name in zip(paths, ["mean", "m2", "count", "frozen"]):
            value = weights.get(path)
            if name in ("mean", "m2"):
                value = value.reshape(-1)  # Keras kept them shaped (1, 1, d)
            put([f"{group}.{name}"], value, path)

    unused = weights.unused()
    if unused:
        raise ConversionError(
            "These weights in the .keras file have no counterpart in the "
            f"converted model, so converting would lose them: {unused}."
        )

    unfilled = [
        k for k in state
        if k not in filled and k not in _PRIOR_BUFFERS_FROM_CONFIG
    ]
    if unfilled:
        if not allow_missing:
            raise ConversionError(
                "The .keras file has no weights for these parts of the model: "
                f"{unfilled}. A file saved before the first training step "
                "(abismal 1.x's epoch_0.keras) holds only the surrogate "
                "posterior; pass allow_missing=True (--allow-missing) to keep "
                "fresh initial values for the rest."
            )
        warnings.warn(
            f"Keeping freshly initialized values for {unfilled}, which the "
            ".keras file does not contain.",
            stacklevel=3,
        )
    model.load_state_dict(state)


def _optimizer(config, weights, model):
    """The 1.x optimizer's class and hyperparameters, with fresh moments, or None."""
    compile_config = config.get("compile_config") or {}
    cfg = compile_config.get("optimizer")
    if not cfg:
        return None
    key = cfg.get("module", "") + "." + cfg.get("class_name", "")
    if key not in _OPTIMIZERS:
        warnings.warn(f"Not converting the optimizer {key}; abismal 2 has no counterpart.", stacklevel=3)
        return None
    c = cfg["config"]
    lr = c.get("learning_rate", 0.001)
    if isinstance(lr, dict):
        # The CLI's --learning-rate-final schedule: record the rate reached,
        # as abismal 2 saves it.
        sched = lr.get("config", {})
        if lr.get("class_name") != "PiecewiseConstantDecay" or "optimizer/vars/0" not in weights:
            warnings.warn(f"Not converting the optimizer: unknown learning rate schedule {lr!r:.200}.", stacklevel=3)
            return None
        step = int(weights.arrays["optimizer/vars/0"])
        bounds, values = sched["boundaries"], sched["values"]
        lr = values[int(np.searchsorted(bounds, step, side="left"))]
    import abismal.optimizers as optimizers
    from abismal.training import parameter_groups
    cls = getattr(optimizers, _OPTIMIZERS[key])
    return cls(
        parameter_groups(model),
        learning_rate=float(lr),
        beta_1=c.get("beta_1", 0.9),
        beta_2=c.get("beta_2", 0.9),
        epsilon=c.get("epsilon", 1e-12),
        weight_decay=c.get("weight_decay"),
        clipnorm=c.get("clipnorm"),
        clipvalue=c.get("clipvalue"),
        global_clipnorm=c.get("global_clipnorm"),
    )


def load_keras(path, allow_missing=False, metadata_length=None):
    """
    Rebuild an abismal 1.x `.keras` model as an abismal 2 model.

    Parameters
    ----------
    path : str
    allow_missing : bool (optional)
        Keep fresh initial values for any part of the model the file has no
        weights for, instead of raising. Needed for 1.x `epoch_0.keras` files,
        which hold only the surrogate posterior.
    metadata_length : int (optional)
        The number of metadata columns. Normally read from the file; required
        only when the file has no trace of it (an unbuilt scale model).

    Returns
    -------
    model : VariationalMergingModel
    optimizer : AbismalOptimizer or None
        The 1.x optimizer's class and hyperparameters on the new model's
        parameters, with fresh moments.
    """
    config, arrays = read_keras(path)
    weights = _Weights(arrays)

    key = _class_key(config)
    if key != "abismal.merging.merging.VariationalMergingModel":
        raise ConversionError(f"{path} holds a {key}, not an abismal VariationalMergingModel.")
    mcfg = {k: v for k, v in config["config"].items() if k not in _KERAS_LAYER_KEYS}
    allowed = {
        "scale_model", "surrogate_posterior", "prior", "likelihood", "mc_samples",
        "kl_weight", "epsilon", "reindexing_ops", "standardize",
        "standardize_intensity", "standardize_metadata",
    }
    unknown = set(mcfg) - allowed
    if unknown:
        raise ConversionError(f"VariationalMergingModel has unknown config keys {sorted(unknown)}.")

    racs = {}
    posterior, posterior_names = _build_posterior(mcfg["surrogate_posterior"], racs)
    prior, prior_names = _build_prior(mcfg["prior"], racs)
    likelihood = _build_likelihood(mcfg["likelihood"])

    scfg = mcfg["scale_model"]
    if _class_key(scfg) != "abismal.scaling.scaling.ImageScaler":
        raise ConversionError(f"Cannot convert scale model {_class_key(scfg)}.")
    scale_kwargs = _layer_config(scfg, _scale_model_keys(), "ImageScaler")
    found = _metadata_length(config, weights)
    if found is None:
        found = metadata_length
    elif metadata_length is not None and metadata_length != found:
        raise ConversionError(f"metadata_length={metadata_length}, but the file has {found} metadata columns.")
    if found is None:
        raise ConversionError(
            f"{path} does not record how many metadata columns the model takes: "
            "its scale model was never built, as in an abismal 1.x "
            "epoch_0.keras, and the file holds the surrogate posterior only. "
            "To convert it anyway, keeping a freshly initialized scale model, "
            "pass allow_missing=True and metadata_length "
            "(--allow-missing --metadata-length N)."
        )
    from abismal.scaling.scaling import ImageScaler
    scale_model = ImageScaler(found, **scale_kwargs)

    from abismal.merging.merging import VariationalMergingModel
    model = VariationalMergingModel(
        scale_model,
        posterior,
        prior,
        likelihood,
        mc_samples=mcfg.get("mc_samples", 1),
        kl_weight=mcfg.get("kl_weight", 1.),
        epsilon=mcfg.get("epsilon", 1e-6),
        reindexing_ops=mcfg.get("reindexing_ops"),
        standardize=mcfg.get("standardize", False),
    )
    for name in ("standardize_intensity", "standardize_metadata"):
        layer = getattr(model, name)
        cfg = mcfg.get(name)
        if layer is None or cfg is None:
            continue
        c = _layer_config(cfg, ("epsilon", "max_counts", "center"), name)
        if "epsilon" in c:
            layer.epsilon = c["epsilon"]
        if c.get("max_counts") is not None:
            layer.max_counts = float(c["max_counts"])

    _fill_state(model, weights, posterior_names, prior_names, allow_missing)
    optimizer = _optimizer(config, weights, model)
    return model, optimizer


def convert_keras(path_in, path_out=None, allow_missing=False, metadata_length=None):
    """
    Convert an abismal 1.x `.keras` model file to an abismal 2 `.pt` file.

    Parameters
    ----------
    path_in : str
    path_out : str (optional)
        Defaults to `path_in` with its `.keras` suffix replaced by `.pt`.
    allow_missing, metadata_length
        See `load_keras`.

    Returns
    -------
    path_out : str
    """
    from abismal.saving import save_model
    path_in = str(path_in)
    if path_out is None:
        stem = path_in[:-len(".keras")] if path_in.endswith(".keras") else path_in
        path_out = stem + ".pt"
    model, optimizer = load_keras(path_in, allow_missing=allow_missing, metadata_length=metadata_length)
    save_model(model, str(path_out), optimizer=optimizer)
    return str(path_out)


def main(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description=(
            "Convert abismal 1.x model files (.keras) to the abismal 2 format "
            "(.pt). The optimizer's settings carry over but not its moments, so "
            "training resumes from fresh Adam moments."
        )
    )
    parser.add_argument("inputs", nargs="+", help=".keras files to convert")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file. Only with a single input; by default each input's "
        ".keras suffix is replaced with .pt.",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Keep fresh initial values for any weights the file lacks, as in "
        "an epoch_0.keras saved before training started.",
    )
    parser.add_argument(
        "--metadata-length", type=int, default=None,
        help="Number of metadata columns, for files whose scale model was never built.",
    )
    parser = parser.parse_args(args)
    if parser.output is not None and len(parser.inputs) > 1:
        raise SystemExit("--output can only be used with a single input file.")
    for path in parser.inputs:
        out = convert_keras(
            path, parser.output,
            allow_missing=parser.allow_missing,
            metadata_length=parser.metadata_length,
        )
        print(f"{path} -> {out}")


if __name__ == "__main__":
    main()
