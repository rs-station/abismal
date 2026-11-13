"""Barron's general robust distribution as the VI model's likelihood (alone, and as the shape of the Ev11 variance)."""
import math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import k1
from abismal import robust
from abismal.likelihood import BarronLikelihood, BarronShape, Ev11Likelihood


def _batch(n=40, mc=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    ipred = torch.rand((n, mc), generator=g) * 4. + 0.5
    iobs = ipred.mean(-1, keepdim=True) + torch.randn((n, 1), generator=g)
    sigiobs = torch.rand((n, 1), generator=g) + 0.2
    return ipred, iobs, sigiobs


def test_robust_is_normal_at_two_cauchy_at_zero_and_normalized_between():
    z = torch.linspace(-6., 6., 49, dtype=torch.float64)
    assert torch.allclose(robust.logpdf(z, 2.), -0.5 * z * z - 0.5 * math.log(2. * math.pi), atol=1e-9)
    assert torch.allclose(robust.logpdf(z, 0.), -torch.log1p(0.5 * z * z) - math.log(math.pi * math.sqrt(2.)), atol=1e-9)
    assert abs(robust.partition(1.) - 2. * math.e * k1(1.)) < 1e-9
    for a in (0.37, 1.234, 1.9):
        f = lambda x: math.exp(float(robust.logpdf(torch.tensor(x, dtype=torch.float64), a)))
        assert abs(2. * (quad(f, 0., 1.)[0] + quad(f, 1., np.inf, limit=500)[0]) - 1.) < 1e-6


def test_barron_at_two_is_the_normal_likelihood():
    ipred, iobs, sigiobs = _batch()
    ll = BarronLikelihood(BarronShape(2.))(ipred, iobs, sigiobs)
    expected = torch.distributions.Normal(iobs, sigiobs).log_prob(ipred)
    assert torch.allclose(ll, expected, atol=1e-5)


def test_a_learned_alpha_starts_where_asked_stays_in_range_and_trains():
    shape = BarronShape(1.5, learn_alpha=True, alpha_min=0.)
    assert float(shape.alpha) == pytest.approx(1.5, abs=1e-5)
    ipred, iobs, sigiobs = _batch()
    BarronLikelihood(shape)(ipred, iobs, sigiobs).sum().backward()
    assert shape.alpha_logit.grad is not None and torch.isfinite(shape.alpha_logit.grad) and shape.alpha_logit.grad != 0.
    with torch.no_grad():
        shape.alpha_logit.fill_(50.)
    assert float(shape.alpha) < 2. and float(shape.alpha) == pytest.approx(2. - shape.ALPHA_MARGIN, abs=1e-6)


def test_a_fixed_alpha_has_nothing_to_train():
    assert len(list(BarronLikelihood(BarronShape(1.)).parameters())) == 0


def test_ev11_with_a_barron_shape_uses_the_ev11_scale():
    ipred, iobs, sigiobs = _batch()
    lik = Ev11Likelihood(a=2., b=0.1, degrees_of_freedom=32., barron=BarronShape(1.))
    sd = torch.sqrt(lik.variance(ipred, sigiobs))
    expected = robust.logpdf((ipred - iobs) / sd, 1.) - torch.log(sd)
    assert torch.allclose(lik(ipred, iobs, sigiobs), expected, atol=1e-6)
    lik(ipred, iobs, sigiobs).sum().backward()
    assert lik.log_a.grad != 0. and lik.log_b.grad != 0.


def test_ev11_refuses_a_barron_shape_outside_the_elbo_fit():
    with pytest.raises(ValueError):
        Ev11Likelihood(degrees_of_freedom=32., fit='normal', barron=BarronShape(1.))
