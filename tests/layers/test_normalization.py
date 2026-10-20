"""The running standardization layers before and after they have seen data."""
import torch

from abismal.layers.normalization import Standardize


def test_with_no_data_seen_standardization_is_the_identity():
    """Moments with count 0 say nothing; dividing by m2/0 = inf used to turn every
    input into exactly zero, whatever it was."""
    x = torch.randn(7, 3)
    for center in (True, False):
        layer = Standardize(3, center=center)
        assert torch.equal(layer(x, training=False), x)
        assert torch.equal(layer.std, torch.ones(3))


def test_once_data_is_seen_it_standardizes():
    torch.manual_seed(0)
    x = 5.0 + 3.0 * torch.randn(10_000, 2)
    layer = Standardize(2)
    out = layer(x, training=True)
    assert torch.allclose(out.mean(0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(out.std(0, correction=0), torch.ones(2), atol=1e-4)


def test_a_frozen_empty_layer_stays_the_identity():
    layer = Standardize(2)
    layer.frozen.fill_(True)
    x = torch.randn(4, 2)
    assert torch.equal(layer(x, training=True), x)
    assert layer.count == 0
