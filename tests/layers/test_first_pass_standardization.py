import numpy as np
import torch
from abismal.layers.normalization import Standardize


def test_standardize_moments_skip_rows_already_counted():
    x = torch.tensor(np.random.default_rng(0).normal(size=(50, 2)), dtype=torch.float32)
    once, repeats = Standardize(2), Standardize(2)
    once.update(x[:30]); once.update(x[30:])
    repeats.update(x[:30])
    # A later batch that repeats those 30 images and adds the other 20: the repeats must not count again.
    rows = torch.tensor([False] * 30 + [True] * 20)
    repeats.update(x, rows)
    assert torch.allclose(repeats.mean, once.mean, atol=1e-5) and torch.allclose(repeats.var, once.var, atol=1e-4)
    assert float(repeats.count) == 50.
    # All rows already seen: nothing moves.
    before = repeats.mean.clone(), repeats.m2.clone()
    repeats.update(x, torch.zeros(50, dtype=torch.bool))
    assert torch.equal(repeats.mean, before[0]) and torch.equal(repeats.m2, before[1]) and float(repeats.count) == 50.
