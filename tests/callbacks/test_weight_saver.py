"""epoch_0.keras is written before the model is built, on purpose.

Its consumer is `abismal.cchalf --sf-init`, which wants the structure factor
initialization. The surrogate posterior is built in its constructor so those
weights are present; the scale model builds lazily and is not. Building it here
would mean pulling a batch through the model before training, filling the
shuffle buffer an extra time -- one of the slowest steps in data loading, paid
twice, for weights nothing reads. So the partial file is intended and Keras'
warning about it is suppressed.
"""

import warnings

import tf_keras as tfk

from abismal.callbacks import WeightSaver


def _unbuilt_model():
    """A model with a weight already, and a layer that has not been built."""
    model = tfk.Sequential([tfk.layers.Dense(3)])
    return model


def test_saving_an_unbuilt_model_emits_no_warning(tmp_path):
    saver = WeightSaver(str(tmp_path))
    saver.model = _unbuilt_model()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        saver.on_train_begin(logs=None)

    unbuilt = [w for w in caught if "has not yet been built" in str(w.message)]
    assert not unbuilt, [str(w.message) for w in unbuilt]
    assert (tmp_path / "epoch_0.keras").exists()


def test_the_file_is_still_written_and_reloadable(tmp_path):
    """Suppressing the warning must not change what lands on disk."""
    saver = WeightSaver(str(tmp_path))
    saver.model = _unbuilt_model()
    saver.on_train_begin(logs=None)

    reloaded = tfk.saving.load_model(str(tmp_path / "epoch_0.keras"))
    assert isinstance(reloaded, tfk.Sequential)


def test_the_suppression_does_not_leak_to_other_saves(tmp_path):
    """A genuine unbuilt-model save elsewhere must still warn.

    The filter is scoped to one call, so anything that follows is unaffected --
    otherwise this would hide a real mistake somewhere else in a run.
    """
    saver = WeightSaver(str(tmp_path))
    saver.model = _unbuilt_model()
    saver.on_train_begin(logs=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tfk.saving.save_model(_unbuilt_model(), str(tmp_path / "elsewhere.keras"))

    assert any("has not yet been built" in str(w.message) for w in caught), (
        "the unbuilt-model warning should still fire outside WeightSaver"
    )
