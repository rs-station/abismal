import numpy as np
import torch

from abismal.distributions import FoldedNormal, Rice, RiceWoolfson


def _params():
    loc = torch.tensor([0.5, 2.0, 3.0, 0.1])
    scale = torch.tensor([1.0, 0.5, 2.0, 1.0])
    centric = torch.tensor([True, False, True, False])
    return loc, scale, centric


def test_selects_the_component_per_element():
    loc, scale, centric = _params()
    q = RiceWoolfson(loc, scale, centric)
    fn, rice = FoldedNormal(loc, scale), Rice(loc, scale)
    x = torch.tensor([0.7, 1.9, 2.5, 0.4])
    for name in ("mean", "variance", "stddev"):
        expected = torch.where(centric, getattr(fn, name), getattr(rice, name))
        np.testing.assert_allclose(getattr(q, name).numpy(), expected.numpy(), rtol=1e-6)
    expected = torch.where(centric, fn.log_prob(x), rice.log_prob(x))
    np.testing.assert_allclose(q.log_prob(x).numpy(), expected.numpy(), rtol=1e-6)


def test_samples_follow_the_selected_moments():
    torch.manual_seed(0)
    loc, scale, centric = _params()
    q = RiceWoolfson(loc, scale, centric)
    z = q.sample((200000,))
    assert z.shape == (200000, 4)
    np.testing.assert_allclose(z.mean(0).numpy(), q.mean.numpy(), rtol=1e-2)
    np.testing.assert_allclose(z.std(0).numpy(), q.stddev.numpy(), rtol=2e-2)


def test_gradients_are_finite():
    loc, scale, centric = _params()
    loc.requires_grad_(True)
    scale.requires_grad_(True)
    q = RiceWoolfson(loc, scale, centric)
    z = q.rsample((16,))
    (q.log_prob(z).sum() + q.mean.sum() + q.stddev.sum()).backward()
    assert torch.isfinite(loc.grad).all() and torch.isfinite(scale.grad).all()


def test_centric_broadcasts():
    q = RiceWoolfson(torch.ones(3, 1), torch.ones(1, 4), torch.tensor([True, False, True, False]))
    assert q.batch_shape == (3, 4)
    assert q.rsample((2,)).shape == (2, 3, 4)
