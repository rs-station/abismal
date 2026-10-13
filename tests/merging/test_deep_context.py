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


@pytest.mark.parametrize("share_weights", [True, False])
@pytest.mark.parametrize("posterior_features", [0, 2])
def test_a_reflection_never_sees_its_own_data(share_weights, posterior_features):
    """Changing one reflection's Iobs, SigIobs or posterior features leaves its own
    scale unchanged and changes the scales of the other reflections on its image."""
    torch.manual_seed(0)
    model = ImageScaler(3, mlp_width=16, mlp_depth=3, prior_name="lognormal", deep_context=True,
                        posterior_features=posterior_features, share_weights=share_weights,
                        batch_normalize=False).eval()
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
    model = ImageScaler(3, mlp_width=8, mlp_depth=2, deep_context=True)
    assert model.get_config()["deep_context"] is True
    assert len(model.context_projections) == 2
    assert not hasattr(ImageScaler(3, mlp_width=8, mlp_depth=2), "context_projections")
