import gemmi
import numpy as np
import pytest
import tensorflow as tf
import tf_keras as tfk

from abismal.symmetry import ReciprocalASU, ReciprocalASUCollection
from abismal.surrogate_posterior.intensity.lognormal import (
    LogNormalPosterior as IntensityLogNormalPosterior,
)
from abismal.surrogate_posterior.structure_factor.lognormal import (
    LogNormalPosterior as StructureFactorLogNormalPosterior,
)


@pytest.fixture(scope="module")
def rac():
    cell = gemmi.UnitCell(30.0, 40.0, 50.0, 90.0, 90.0, 90.0)
    sg = gemmi.SpaceGroup("P 21 21 21")
    return ReciprocalASUCollection(ReciprocalASU(cell, sg, 8.0, anomalous=True))


@pytest.fixture(scope="module")
def moments(rac):
    """Requested mean and standard deviation, spanning a wide range of shapes."""
    rng = np.random.default_rng(1234)
    mean = rng.uniform(0.3, 3.0, rac.asu_size).astype("float32")
    stddev = rng.uniform(0.05, 1.5, rac.asu_size).astype("float32")
    return mean, stddev


@pytest.mark.parametrize(
    "Posterior",
    [StructureFactorLogNormalPosterior, IntensityLogNormalPosterior],
)
def test_init_matches_the_requested_moments(rac, moments, Posterior):
    """loc_init/scale_init are moments, not the log normal's own parameters."""
    mean, stddev = moments
    q = Posterior(rac, loc_init=mean, scale_init=stddev).flat_distribution()
    assert np.allclose(q.mean().numpy(), mean, rtol=1e-5)
    assert np.allclose(q.stddev().numpy(), stddev, rtol=1e-5)


@pytest.mark.parametrize(
    "Posterior",
    [StructureFactorLogNormalPosterior, IntensityLogNormalPosterior],
)
def test_default_init_is_well_conditioned(rac, Posterior):
    q = Posterior(rac).flat_distribution()
    for p in (q.loc, q.scale):
        assert np.all(np.isfinite(p.numpy()))
    # The unconstrained scale sits near log(0.01), not down at log(epsilon),
    # where gradients would be dead.
    assert np.all(q.scale.numpy() > 1e-4)


def test_structure_factor_isigi_is_exact(rac, moments):
    """F log normal implies F**2 log normal, so I moments are closed form."""
    mean, stddev = moments
    q = StructureFactorLogNormalPosterior(rac, loc_init=mean, scale_init=stddev)
    I, SIGI = (x.numpy() for x in q.get_flat_isigi())

    z = q.flat_distribution().sample(200000, seed=0) ** 2
    # Fourth moments of a log normal converge slowly, so compare only where the
    # tail is mild enough for the Monte Carlo estimate to be trustworthy.
    keep = (stddev / mean) < 0.5
    assert keep.sum() > 10
    assert np.allclose(I[keep], tf.reduce_mean(z, 0).numpy()[keep], rtol=0.05)
    assert np.allclose(SIGI[keep], tf.math.reduce_std(z, 0).numpy()[keep], rtol=0.10)


def test_intensity_fsigf_is_exact(rac, moments):
    """The square root of a log normal is log normal, so F moments are too."""
    mean, stddev = moments
    q = IntensityLogNormalPosterior(rac, loc_init=mean, scale_init=stddev)
    F, SIGF = (x.numpy() for x in q.get_flat_fsigf())

    z = tf.sqrt(q.flat_distribution().sample(200000, seed=0))
    assert np.allclose(F, tf.reduce_mean(z, 0).numpy(), rtol=0.02)
    assert np.allclose(SIGF, tf.math.reduce_std(z, 0).numpy(), rtol=0.05)


def test_squaring_beats_the_inherited_approximation(rac, moments):
    """The override is not cosmetic: the base class propagation is biased.

    StructureFactorPosteriorBase estimates I as F**2 + SIGF**2, which is exact
    for the mean but leaves SIGI to first order. Check the analytic SIGI
    actually differs, so a regression that dropped the override would show up.
    """
    mean, stddev = moments
    q = StructureFactorLogNormalPosterior(rac, loc_init=mean, scale_init=stddev)
    _, exact = q.get_flat_isigi()
    F, SIGF = q.get_flat_fsigf()
    propagated = np.abs(2.0 * F.numpy() * SIGF.numpy())
    assert not np.allclose(exact.numpy(), propagated, rtol=0.01)


@pytest.mark.parametrize(
    "Posterior",
    [StructureFactorLogNormalPosterior, IntensityLogNormalPosterior],
)
def test_both_parameters_are_trainable(rac, Posterior):
    """The lazy optimizer keys off trainable_variables, so both must be there."""
    q = Posterior(rac)
    assert len(q.trainable_variables) == 2

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(q.flat_distribution().log_prob(tf.ones(rac.asu_size)))
    grads = tape.gradient(loss, q.trainable_variables)
    assert all(g is not None for g in grads)
    assert all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads)


@pytest.mark.parametrize(
    "Posterior",
    [StructureFactorLogNormalPosterior, IntensityLogNormalPosterior],
)
def test_round_trips_through_keras(rac, moments, Posterior, tmp_path):
    mean, stddev = moments
    q = Posterior(rac, loc_init=mean, scale_init=stddev)

    path = str(tmp_path / "posterior.keras")
    q.save(path)
    reloaded = tfk.saving.load_model(path)

    assert type(reloaded) is Posterior
    assert reloaded.rac.asu_size == rac.asu_size
    before, after = q.flat_distribution(), reloaded.flat_distribution()
    assert np.array_equal(before.loc.numpy(), after.loc.numpy())
    assert np.array_equal(before.scale.numpy(), after.scale.numpy())


def test_gather_selects_the_right_reflections(rac, moments):
    """distribution() and flat_distribution() must agree reflection by reflection."""
    mean, stddev = moments
    q = StructureFactorLogNormalPosterior(rac, loc_init=mean, scale_init=stddev)

    H = rac.reciprocal_asus[0].Hunique[:16]
    asu_id = tf.zeros((len(H), 1), dtype="int32")
    gathered = q.distribution(asu_id, tf.convert_to_tensor(H))
    flat = q.flat_distribution()

    assert np.allclose(gathered.mean().numpy(), flat.mean().numpy()[: len(H)])


def test_registered_as_a_cli_choice():
    from abismal.command_line.parser import parser

    args = parser.parse_args([
        "-d", "1.8",
        "--posterior-distribution", "lognormal",
        "-o", "unused",
        "unused.mtz",
    ])
    assert args.posterior_distribution == "lognormal"
