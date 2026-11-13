"""The Ev11 likelihood's variance forms: the linear 'photon' term."""
import pytest
import torch
from abismal.likelihood import Ev11Likelihood


def _batch(n=50, mc=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand((n, 1), generator=g) * 3. + 0.05
    ipred = s * (torch.rand((n, mc), generator=g) * 5. + 0.1)
    iobs = torch.randn((n, 1), generator=g) + ipred.mean(-1, keepdim=True)
    sigiobs = torch.rand((n, 1), generator=g) + 0.1
    return ipred, iobs, sigiobs


def test_the_photon_form_adds_c_times_the_intensity():
    ipred, iobs, sigiobs = _batch()
    lik = Ev11Likelihood(a=2., b=0.1, c=0.3, form='photon')
    p = ipred.mean(-1, keepdim=True)
    expected = 2. * sigiobs ** 2 + (0.1 * p) ** 2 + 0.3 * p
    assert torch.allclose(lik.variance(ipred, sigiobs), expected, rtol=1e-5)


def test_the_photon_form_parameters_train():
    ipred, iobs, sigiobs = _batch()
    lik = Ev11Likelihood(a=2., b=0.1, degrees_of_freedom=32., form='photon')
    lik(ipred, iobs, sigiobs).sum().backward()
    names = ['log_a', 'log_b', 'log_c']
    for name in names:
        g = getattr(lik, name).grad
        assert g is not None and torch.isfinite(g) and g != 0.
    trainable = {n for n, p in lik.named_parameters() if p.requires_grad}
    assert trainable == set(names)


def test_the_photon_form_refuses_the_online_fit():
    with pytest.raises(ValueError):
        Ev11Likelihood(form='photon', fit='online')


def test_an_unknown_form_is_refused():
    with pytest.raises(ValueError):
        Ev11Likelihood(form='aimless')
