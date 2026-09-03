import numpy as np
import pytest
import tensorflow as tf
import tf_keras as tfk

from abismal.layers import FourierFeatures


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(1234)
    return rng.standard_normal((300, 6)).astype("float32")


def sqdist(X):
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


@pytest.mark.parametrize("length_scale", [0.25, 0.5, 1.0, 2.0])
def test_induced_kernel_matches_the_declared_length_scale(points, length_scale):
    """The reason the layer exists: z(x).z(y) estimates a Gaussian kernel.

    That is what pins how fast a network reading these features can vary.
    """
    z = FourierFeatures(4096, length_scale=length_scale, seed=0)(points).numpy()
    expected = np.exp(-sqdist(points) / (2 * length_scale**2))
    assert np.abs(z @ z.T - expected).max() < 0.1


def test_kernel_error_falls_off_as_one_over_sqrt_m(points):
    expected = np.exp(-sqdist(points) / 2)

    def rms(m):
        z = FourierFeatures(m, length_scale=1.0, seed=1)(points).numpy()
        return np.sqrt(((z @ z.T - expected) ** 2).mean())

    coarse, fine = rms(256), rms(4096)
    # 16x the frequencies should cut the error by roughly 4x.
    assert 2.5 < coarse / fine < 6.0


def test_per_column_length_scale_gives_an_anisotropic_kernel():
    """A vector length scale gives each input direction its own bandwidth."""
    layer = FourierFeatures(4096, length_scale=[0.2, 5.0], seed=0)
    origin = np.zeros((1, 2), dtype="float32")
    along_tight = np.array([[1.0, 0.0]], dtype="float32")
    along_loose = np.array([[0.0, 1.0]], dtype="float32")

    z0 = layer(origin).numpy()
    k_tight = (z0 @ layer(along_tight).numpy().T).item()
    k_loose = (z0 @ layer(along_loose).numpy().T).item()

    # One unit along the tight axis is 5 length scales; along the loose axis
    # it is a fifth of one.
    assert k_tight < 0.01
    assert k_loose > 0.95


def test_output_width_is_two_per_frequency_and_excludes_the_raw_input(points):
    layer = FourierFeatures(32, seed=0)
    out = layer(points)
    assert out.shape == (len(points), 64)
    assert layer.compute_output_shape(points.shape) == (len(points), 64)
    # Features are bounded by construction; a passed-through raw coordinate
    # would not be.
    assert np.abs(out.numpy()).max() <= 1.0 / np.sqrt(32) + 1e-6


def test_B_is_not_trainable_by_default(points):
    layer = FourierFeatures(16, seed=0)
    layer(points)
    assert len(layer.trainable_weights) == 0
    assert len(layer.non_trainable_weights) == 1

    trainable = FourierFeatures(16, seed=0, trainable_B=True)
    trainable(points)
    assert len(trainable.trainable_weights) == 1


def test_encoding_is_fixed_across_calls(points):
    """A resampled B would be a different function every step."""
    layer = FourierFeatures(64, seed=0)
    assert np.array_equal(layer(points).numpy(), layer(points).numpy())


def test_seed_controls_the_draw(points):
    a = FourierFeatures(64, seed=7)(points).numpy()
    b = FourierFeatures(64, seed=7)(points).numpy()
    c = FourierFeatures(64, seed=8)(points).numpy()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_round_trips_through_keras(points, tmp_path):
    """B has to survive saving: a redrawn one is a different function."""
    model = tfk.Sequential([
        tfk.layers.InputLayer(input_shape=(points.shape[-1],)),
        FourierFeatures(64, length_scale=0.5, seed=3),
        tfk.layers.Dense(4),
    ])
    before = model(points).numpy()

    path = str(tmp_path / "ff.keras")
    model.save(path)
    reloaded = tfk.saving.load_model(path)

    assert np.allclose(before, reloaded(points).numpy(), atol=1e-6)
    layer = reloaded.layers[0]
    assert layer.length_scale == 0.5
    assert layer.num_frequencies == 64
    assert layer.trainable_B is False


def test_accepts_ragged_input_like_dense_does(points):
    """It stands in for a Dense on ragged per-image batches, so it must take one."""
    layer = FourierFeatures(16, seed=0)
    splits = tf.constant([0, 100, 300], dtype=tf.int64)
    ragged = tf.RaggedTensor.from_row_splits(tf.convert_to_tensor(points), splits)

    out = layer(ragged)
    assert isinstance(out, tf.RaggedTensor)
    assert out.shape[0] == 2
    assert out.flat_values.shape == (len(points), 32)
    assert np.allclose(out.flat_values.numpy(), layer(points).numpy())

    # and going through map_flat_values explicitly agrees
    viaflat = tf.ragged.map_flat_values(layer, ragged)
    assert np.allclose(viaflat.flat_values.numpy(), out.flat_values.numpy())


@pytest.mark.parametrize("bad", [0.0, -1.0, [1.0, -2.0]])
def test_non_positive_length_scale_is_rejected(bad):
    n = 2 if isinstance(bad, list) else 6
    with pytest.raises(ValueError, match="length_scale must be positive"):
        FourierFeatures(8, length_scale=bad, seed=0)(np.zeros((3, n), "float32"))


def test_a_longer_length_scale_is_smoother(points):
    """Behavioural restatement of the kernel property."""
    d = np.sqrt(sqdist(points))
    near = d < np.percentile(d[d > 0], 5)

    def mean_kernel(ell):
        z = FourierFeatures(2048, length_scale=ell, seed=0)(points).numpy()
        return (z @ z.T)[near].mean()

    assert mean_kernel(0.1) < mean_kernel(1.0) < mean_kernel(10.0)
