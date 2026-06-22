from os.path import abspath, dirname, join

import numpy as np
import pytest
import tensorflow as tf

from abismal.io.manager import DataManager
from abismal.symmetry import ReciprocalASU, ReciprocalASUGraph
from abismal.prior.intensity.wilson import WilsonPrior
from reciprocalspaceship.utils import bin_by_percentile


@pytest.fixture(scope="module")
def empirical_inputs():
    """Build a (rac, batches) pair from the conventional MTZ fixture.

    The batches are materialized into a concrete list so that
    ``with_empirical_sigma`` and the reference estimator below iterate over
    *exactly* the same data in the same order (the streaming accumulator's
    correctness only makes sense relative to a fixed sequence of batches).
    A batch size of 1 splits the handful of images in the fixture across
    several batches, which is what exercises the batch-to-batch Welford
    combine step.
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
    assert len(batches) > 1, "need multiple batches to exercise the streaming combine"

    rasu = [
        ReciprocalASU(dm.cell, dm.spacegroup, dm.dmin, anomalous=True)
        for _ in range(dm.num_asus)
    ]
    rac = ReciprocalASUGraph(*rasu)
    return rac, batches


def _direct_weighted_sigma(rac, batches, bins, isigi_cutoff, weighted=True):
    """Reference (non-welfordized) per-bin estimator.

    Accumulates the totals ``Σ w·I`` and ``Σ w`` across *all* observations in a
    single pass, then divides once at the end -- i.e. no running/streaming
    update. With ``weighted=True`` the weights are the inverse variances
    ``w = mask / σ²`` that ``with_empirical_sigma`` uses; with ``weighted=False``
    every (unmasked) observation gets unit weight. Returns the per-reflection
    sigma array (``mean`` gathered back through the resolution-bin labels), to
    match the ``standardize=False, interpolate=False`` output of the method.
    """
    labels, _ = bin_by_percentile(rac.dHKL, bins=bins, ascending=False)
    num = np.zeros(bins)
    den = np.zeros(bins)
    for x, _y in batches:
        asu_id, hkl, _res, _wav, _meta, iobs, sigiobs = x
        idx = rac.gather(labels, asu_id, hkl).flat_values.numpy()
        I = np.squeeze(iobs.flat_values.numpy(), axis=-1)
        SIGI = np.squeeze(sigiobs.flat_values.numpy(), axis=-1)

        if isigi_cutoff is not None:
            mask = (I / SIGI > isigi_cutoff).astype("float32")
        else:
            mask = np.ones_like(I)

        w = mask / SIGI**2 if weighted else mask
        np.add.at(num, idx, w * I)
        np.add.at(den, idx, w)

    mean = (num / den).astype("float32")
    return tf.gather(mean, labels).numpy()


@pytest.mark.parametrize("bins", [4, 6, 8, 10])
@pytest.mark.parametrize("isigi_cutoff", [None, 1.0])
def test_welford_matches_direct_weighted_mean(empirical_inputs, bins, isigi_cutoff):
    """The streaming inverse-variance Welford accumulator must reproduce the
    weighted mean computed directly (accumulate-then-divide) over the same data."""
    rac, batches = empirical_inputs

    prior = WilsonPrior.with_empirical_sigma(
        rac,
        batches,
        bins=bins,
        isigi_cutoff=isigi_cutoff,
        standardize=False,
        interpolate=False,
    )
    got = np.asarray(prior.sigma)

    expected = _direct_weighted_sigma(rac, batches, bins, isigi_cutoff)

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_inverse_variance_weighting_is_applied(empirical_inputs):
    """Guard against a regression that drops the weights: with non-constant
    sigmas the inverse-variance-weighted result must differ from the plain
    (unweighted) per-bin mean."""
    rac, batches = empirical_inputs
    bins = 8

    prior = WilsonPrior.with_empirical_sigma(
        rac,
        batches,
        bins=bins,
        isigi_cutoff=None,
        standardize=False,
        interpolate=False,
    )
    weighted = np.asarray(prior.sigma)

    unweighted = _direct_weighted_sigma(
        rac, batches, bins, isigi_cutoff=None, weighted=False
    )

    # Sanity: the method agrees with the *weighted* reference...
    expected = _direct_weighted_sigma(rac, batches, bins, isigi_cutoff=None)
    np.testing.assert_allclose(weighted, expected, rtol=1e-4)

    # ...but is meaningfully different from the unweighted estimator.
    assert not np.allclose(weighted, unweighted, rtol=1e-2)


def _make_batch(hkl, I, SIGI):
    """Hand-build one (x, y) batch (a single image) of the ragged shape the
    loader emits, from plain numpy arrays of miller indices / intensities."""
    n = len(hkl)

    def ragged(values):
        return tf.RaggedTensor.from_tensor(values[None], ragged_rank=1)

    asu_id = ragged(tf.zeros((n, 1), tf.int32))
    hkl_rt = ragged(tf.cast(hkl, tf.int32))
    dHKL = ragged(tf.ones((n, 1), tf.float32))
    wavelength = ragged(tf.ones((n, 1), tf.float32))
    metadata = ragged(tf.ones((n, 1), tf.float32))
    iobs = ragged(tf.cast(I, tf.float32)[:, None])
    sigiobs = ragged(tf.cast(SIGI, tf.float32)[:, None])
    x = (asu_id, hkl_rt, dHKL, wavelength, metadata, iobs, sigiobs)
    return x, (iobs,)


def test_empty_bin_in_a_batch_stays_finite(empirical_inputs):
    """A resolution bin that is absent from an individual batch must not poison
    the running mean with NaN; the final estimate must still match the direct
    weighted mean over the union of all batches."""
    rac, _ = empirical_inputs
    bins = 8

    # Split the ASU's reflections into two contiguous resolution ranges. Each
    # batch then populates only a subset of the percentile bins (leaving the
    # rest empty within that batch), while the union covers every bin.
    H = np.asarray(rac.Hunique)
    order = np.argsort(np.asarray(rac.dHKL))
    H = H[order]
    half = len(H) // 2
    rng = np.random.default_rng(0)

    batches = []
    for Hpart in (H[:half], H[half:]):
        m = len(Hpart)
        I = rng.uniform(1.0, 100.0, m)
        SIGI = rng.uniform(0.1, 10.0, m)  # non-constant -> weights actually vary
        batches.append(_make_batch(Hpart, I, SIGI))

    # Confirm the test is actually meaningful: at least one batch leaves a bin
    # empty (which is exactly the condition the guard protects against).
    labels, _ = bin_by_percentile(rac.dHKL, bins=bins, ascending=False)
    empties = []
    for x, _y in batches:
        idx = rac.gather(labels, x[0], x[1]).flat_values.numpy()
        empties.append((np.bincount(idx, minlength=bins) == 0).any())
    assert any(empties)

    prior = WilsonPrior.with_empirical_sigma(
        rac,
        batches,
        bins=bins,
        isigi_cutoff=None,
        standardize=False,
        interpolate=False,
    )
    got = np.asarray(prior.sigma)

    assert np.all(np.isfinite(got))
    expected = _direct_weighted_sigma(rac, batches, bins, isigi_cutoff=None)
    np.testing.assert_allclose(got, expected, rtol=1e-4)
