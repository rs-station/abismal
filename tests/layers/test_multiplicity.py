from os.path import abspath, dirname, join

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as tfk

from abismal.io.manager import DataManager
from abismal.layers.normalization import Multiplicity
from abismal.symmetry import ReciprocalASU, ReciprocalASUGraph


@pytest.fixture(scope="module")
def rac_and_batches():
    """A (rac, batches) pair built from the conventional MTZ fixture.

    Batches are materialized into a list so every test iterates over exactly the
    same observations in the same order -- the running counts only make sense
    relative to a fixed sequence. batch_size=1 splits the fixture's images
    across several batches, which is what exercises accumulation.
    """
    conventional_mtz = abspath(join(dirname(__file__), "..", "data", "conventional.mtz"))
    dm = DataManager(
        [conventional_mtz],
        dmin=4.0,
        num_cpus=1,
        wavelength=1.0,
        test_fraction=0.0,
        isigi_cutoff=None,
        shuffle_buffer_size=0,
        batch_size=1,
    )
    train, _ = dm.get_train_test_splits()
    batches = list(train)
    assert len(batches) > 1, "need multiple batches to exercise accumulation"

    rasu = [
        ReciprocalASU(dm.cell, dm.spacegroup, dm.dmin, anomalous=True)
        for _ in range(dm.num_asus)
    ]
    rac = ReciprocalASUGraph(*rasu)
    return rac, batches


def _reference_counts(rac, batches):
    """Direct per-reflection observation counts, computed without the layer."""
    counts = np.zeros(rac.asu_size, dtype="float64")
    for x, _y in batches:
        asu_id, hkl = x[0], x[1]
        hid = rac._miller_ids(asu_id.flat_values, hkl.flat_values).numpy()
        hid = hid[hid >= 0]
        np.add.at(counts, hid, 1.0)
    return counts


def test_counts_match_direct_bincount(rac_and_batches):
    """Accumulated counts equal a direct count, offset by the pseudocount."""
    rac, batches = rac_and_batches
    layer = Multiplicity(rac)

    # counts are initialized to one, not zero
    assert np.allclose(layer.counts.numpy(), 1.0)

    for x, _y in batches:
        layer(x, training=True)

    expected = _reference_counts(rac, batches) + 1.0
    assert np.allclose(layer.counts.numpy(), expected)
    # the fixture must actually contain repeated observations, or the test is vacuous
    assert expected.max() > 2.0


def test_weight_shape_broadcasts_against_log_likelihood(rac_and_batches):
    """Weights are ragged [batch, None, 1] and broadcast against [.., mc_samples]."""
    rac, batches = rac_and_batches
    layer = Multiplicity(rac)
    x = batches[0][0]
    iobs = x[5]

    w = layer(x, training=True)

    assert isinstance(w, tf.RaggedTensor)
    assert w.flat_values.shape[-1] == 1
    assert w.flat_values.shape[0] == iobs.flat_values.shape[0]
    assert np.array_equal(w.row_splits.numpy(), iobs.row_splits.numpy())

    mc_samples = 7
    ll = tf.ragged.map_flat_values(
        lambda v: tf.random.uniform(tf.concat([tf.shape(v)[:1], [mc_samples]], axis=0)),
        iobs,
    )
    weighted = ll * w
    assert weighted.flat_values.shape == (iobs.flat_values.shape[0], mc_samples)


def test_weights_have_unit_mean(rac_and_batches):
    rac, batches = rac_and_batches
    layer = Multiplicity(rac)
    for x, _y in batches:
        w = layer(x, training=True)
        assert np.isclose(np.mean(w.flat_values.numpy()), 1.0, atol=1e-5)


def test_weights_track_multiplicity(rac_and_batches):
    """Weight ratios equal multiplicity ratios for the observations in a batch."""
    rac, batches = rac_and_batches
    layer = Multiplicity(rac)
    for x, _y in batches:
        layer(x, training=True)

    # freeze so the final call does not perturb the counts it is compared against
    layer.frozen.assign(True)
    x = batches[0][0]
    w = layer(x, training=True).flat_values.numpy().squeeze(-1)

    counts = layer.counts.numpy()
    hid = rac._miller_ids(x[0].flat_values, x[1].flat_values).numpy()
    expected = counts[np.maximum(hid, 0)]
    expected = expected / expected.mean()

    assert np.allclose(w, expected, rtol=1e-5)


@pytest.mark.parametrize("exponent", [1.0, -1.0, 0.5])
def test_exponent_applied_before_normalization(rac_and_batches, exponent):
    rac, batches = rac_and_batches
    layer = Multiplicity(rac, exponent=exponent)
    for x, _y in batches:
        layer(x, training=True)

    layer.frozen.assign(True)
    x = batches[0][0]
    w = layer(x, training=True).flat_values.numpy().squeeze(-1)

    counts = layer.counts.numpy()
    hid = rac._miller_ids(x[0].flat_values, x[1].flat_values).numpy()
    expected = counts[np.maximum(hid, 0)] ** exponent
    expected = expected / expected.mean()

    assert np.allclose(w, expected, rtol=1e-5)
    assert np.all(np.isfinite(w))


def test_exponent_zero_gives_uniform_weights(rac_and_batches):
    """exponent=0 disables the reweighting."""
    rac, batches = rac_and_batches
    layer = Multiplicity(rac, exponent=0.0)
    for x, _y in batches:
        w = layer(x, training=True)
        assert np.allclose(w.flat_values.numpy(), 1.0)


def test_frozen_stops_counting(rac_and_batches):
    rac, batches = rac_and_batches
    layer = Multiplicity(rac)

    for x, _y in batches:
        layer(x, training=True)
    before = layer.counts.numpy().copy()

    layer.frozen.assign(True)
    for x, _y in batches:
        layer(x, training=True)

    assert np.array_equal(layer.counts.numpy(), before)


def test_no_counting_outside_training(rac_and_batches):
    """Validation passes must not contribute to the multiplicity."""
    rac, batches = rac_and_batches
    layer = Multiplicity(rac)

    for x, _y in batches:
        layer(x, training=False)
    assert np.allclose(layer.counts.numpy(), 1.0)

    for x, _y in batches:
        layer(x)
    assert np.allclose(layer.counts.numpy(), 1.0)


def test_out_of_asu_millers_are_not_counted(rac_and_batches):
    """Millers outside the ASU map to -1 and must not inflate reflection 0."""
    rac, _batches = rac_and_batches
    layer = Multiplicity(rac)

    hmax = np.asarray(rac.hmax)
    bogus = (hmax + 50)[None, :].astype("int32")
    asu_id = tf.RaggedTensor.from_row_lengths(tf.zeros((1, 1), dtype="int32"), [1])
    hkl = tf.RaggedTensor.from_row_lengths(tf.convert_to_tensor(bogus), [1])
    inputs = (asu_id, hkl, None, None, None, None, None)

    layer(inputs, training=True)
    assert np.allclose(layer.counts.numpy(), 1.0)


def test_serialization_round_trip(rac_and_batches):
    rac, batches = rac_and_batches
    layer = Multiplicity(rac, exponent=-1.0)
    for x, _y in batches:
        layer(x, training=True)

    config = layer.get_config()
    restored = Multiplicity.from_config(config)

    assert restored.exponent == layer.exponent
    assert restored.rac.asu_size == rac.asu_size
    # counts are layer weights, not config, so they come back at the initializer
    assert np.allclose(restored.counts.numpy(), 1.0)

    restored.counts.assign(layer.counts)
    restored.frozen.assign(True)
    layer.frozen.assign(True)
    x = batches[0][0]
    assert np.allclose(
        layer(x, training=True).flat_values.numpy(),
        restored(x, training=True).flat_values.numpy(),
    )
