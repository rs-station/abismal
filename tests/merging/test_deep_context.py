"""--deep-context: the image track is pooled after every layer into the scale track."""
import numpy as np
import pytest
import torch

from abismal.io import Batch
from abismal.scaling import ImageScaler


def batch(num_images=3, refls_per_image=12, m=3, seed=0):
    rng = np.random.default_rng(seed)
    n = num_images * refls_per_image
    I = rng.gamma(1., 10., size=(n, 1)).astype(np.float32)
    return Batch(
        torch.zeros((n, 1), dtype=torch.int32),
        torch.as_tensor(rng.integers(-5, 5, size=(n, 3)), dtype=torch.int32),
        torch.ones((n, 1)),
        torch.ones((n, 1)),
        torch.as_tensor(rng.normal(size=(n, m)), dtype=torch.float32),
        torch.as_tensor(I),
        torch.as_tensor(np.sqrt(I + 1.)),
        torch.as_tensor(np.repeat(np.arange(num_images), refls_per_image)),
        num_images,
    )


def scales(model, b, features=None, seed=1):
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(b, mc_samples=8, training=False, features=features)


@pytest.mark.parametrize("mode", ["projected", "raw", "delta"])
@pytest.mark.parametrize("share_weights", [True, False])
@pytest.mark.parametrize("posterior_features", [0, 2])
def test_a_reflection_never_sees_its_own_data(mode, share_weights, posterior_features):
    """Changing one reflection's Iobs, SigIobs or posterior features leaves its own
    scale unchanged and changes the scales of the other reflections on its image."""
    torch.manual_seed(0)
    model = ImageScaler(3, mlp_width=16, mlp_depth=3, prior_name="lognormal", deep_context=mode,
                        posterior_features=posterior_features, share_weights=share_weights,
                        batch_normalize=False).eval()
    # Move the zero-initialized output layers of the blocks off zero, so that
    # every layer's increment, and hence every pool, carries information
    with torch.no_grad():
        for block in list(model.image_network) + list(model.scale_network):
            block.ff2.weight.normal_(0., 0.3)
    b = batch()
    features = torch.rand(b.iobs.shape[0], 2) if posterior_features else None
    before = scales(model, b, features)
    i = 5  # on image 0, which holds rows 0-11
    iobs = b.iobs.clone()
    iobs[i] = iobs[i] * 10. + 100.
    sigiobs = b.sigiobs.clone()
    sigiobs[i] = sigiobs[i] * 3.
    if features is not None:
        features = features.clone()
        features[i] = features[i] + 5.
    after = scales(model, b._replace(iobs=iobs, sigiobs=sigiobs), features)
    torch.testing.assert_close(after[i], before[i])
    others = [j for j in range(12) if j != i]
    assert not torch.allclose(after[others], before[others])
    torch.testing.assert_close(after[12:], before[12:])  # other images are untouched


def test_config_round_trip():
    model = ImageScaler(3, mlp_width=8, mlp_depth=2, deep_context=True)  # old configs stored a bool
    assert model.get_config()["deep_context"] == "projected"
    assert len(model.context_projections) == 2
    for mode in ("raw", "delta"):
        m = ImageScaler(3, mlp_width=8, mlp_depth=2, deep_context=mode)
        assert m.get_config()["deep_context"] == mode
        assert not hasattr(m, "context_projections")
    assert ImageScaler(3, mlp_width=8, mlp_depth=2).get_config()["deep_context"] is None
    with pytest.raises(ValueError):
        ImageScaler(3, mlp_width=8, mlp_depth=2, deep_context="sideways")


def test_delta_matches_single_pooling_at_initialization():
    """Every block starts as the identity, so all increments are zero and the delta mode
    hands the scale track exactly pool(h_0) = pool(h_L), as single pooling does."""
    b = batch()
    single = ImageScaler(3, mlp_width=16, mlp_depth=4, prior_name="lognormal", batch_normalize=False).eval()
    delta = ImageScaler(3, mlp_width=16, mlp_depth=4, prior_name="lognormal", batch_normalize=False,
                        deep_context="delta").eval()
    delta.load_state_dict(single.state_dict())
    torch.testing.assert_close(scales(delta, b), scales(single, b))


def test_branch_is_the_residual():
    from abismal.layers.feed_forward import FeedForward, GLUFeedForward
    for FF in (FeedForward, GLUFeedForward):
        f = FF(8, pre_activation="l2", activation="swish", epsilon=0.)
        with torch.no_grad():
            f.build(lambda w: torch.nn.init.normal_(w))
            for p in f.parameters():
                p.normal_()
        x = torch.randn(5, 8)
        torch.testing.assert_close(f(x), x + f.branch(x))
