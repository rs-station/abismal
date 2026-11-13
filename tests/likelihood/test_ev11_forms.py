"""The Ev11 likelihood's variance forms: the linear 'photon' and 'gscaled' terms and Aimless's SDcorrection."""
import math
import pytest
import torch
from abismal.likelihood import Ev11Likelihood


def _batch(n=50, mc=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand((n, 1), generator=g) * 3. + 0.05
    ipred = s * (torch.rand((n, mc), generator=g) * 5. + 0.1)
    iobs = torch.randn((n, 1), generator=g) + ipred.mean(-1, keepdim=True)
    sigiobs = torch.rand((n, 1), generator=g) + 0.1
    return ipred, iobs, sigiobs, s


def _sigma_prime(sigma, g, iav, sdfac, sdb, sdadd):
    """Aimless sdctypes.cpp SDcorrection::SigmaPrime, transcribed (scalars), with SetParameters's limits."""
    sdfac = max(sdfac, 0.2)
    sdadd = max(-1., min(1., sdadd))
    sdadd2 = -sdadd * sdadd if sdadd < 0. else sdadd * sdadd
    var = sigma * sigma
    gi = g * iav
    ssc = max(0.1 * var, var + sdb * g * gi + sdadd2 * gi * gi)
    return sdfac * math.sqrt(ssc)


def test_the_aimless_form_starts_where_the_ev11_form_does():
    ipred, iobs, sigiobs, s = _batch()
    ev11 = Ev11Likelihood(a=2., b=0.1)
    aimless = Ev11Likelihood(a=2., b=0.1, form='aimless')
    assert torch.allclose(ev11.variance(ipred, sigiobs), aimless.variance(ipred, sigiobs, s), rtol=1e-5)


@pytest.mark.parametrize('sdfac,sdb,sdadd', [(1.3, 0.5, 0.2), (0.1, 0.0, 0.05), (1.0, -40., 0.1), (0.9, 0.3, -0.3),
                                             (1.1, 0.0, 2.5)])
def test_the_aimless_form_is_aimless_sigma_prime(sdfac, sdb, sdadd):
    """Including its limits: SdFac floored at 0.2, |SdAdd| capped at 1, the bracket floored at 0.1 sigma^2."""
    ipred, iobs, sigiobs, s = _batch()
    lik = Ev11Likelihood(form='aimless')
    with torch.no_grad():
        lik.sdfac.fill_(sdfac); lik.sdb.fill_(sdb); lik.sdadd.fill_(sdadd)
    var = lik.variance(ipred, sigiobs, s).squeeze(-1)
    iav = (ipred / s).mean(-1)  # the scaled intensity <I>, so that g*<I> is the mean prediction
    expected = torch.tensor([_sigma_prime(float(sg), float(g), float(i), sdfac, sdb, sdadd) ** 2
                             for sg, g, i in zip(sigiobs.squeeze(-1), s.squeeze(-1), iav)])
    assert torch.allclose(var, expected, rtol=1e-4)


def test_the_aimless_floor_binds_only_when_the_bracket_falls_below_a_tenth_of_sigma_squared():
    ipred, iobs, sigiobs, s = _batch()
    lik = Ev11Likelihood(form='aimless')
    with torch.no_grad():
        lik.sdb.fill_(-1e3)
    assert torch.allclose(lik.variance(ipred, sigiobs, s), 0.1 * sigiobs ** 2)


@pytest.mark.parametrize('form', ['photon', 'gscaled'])
def test_the_linear_forms_add_c_times_the_raw_or_scaled_intensity(form):
    ipred, iobs, sigiobs, s = _batch()
    lik = Ev11Likelihood(a=2., b=0.1, c=0.3, form=form)
    p = ipred.mean(-1, keepdim=True)
    linear = p if form == 'photon' else s * p
    expected = 2. * sigiobs ** 2 + (0.1 * p) ** 2 + 0.3 * linear
    assert torch.allclose(lik.variance(ipred, sigiobs, s), expected, rtol=1e-5)


def test_the_scaled_forms_need_the_scale():
    ipred, iobs, sigiobs, s = _batch()
    assert not Ev11Likelihood(form='photon').needs_scale
    for form in ('gscaled', 'aimless'):
        lik = Ev11Likelihood(form=form)
        assert lik.needs_scale
        with pytest.raises(ValueError):
            lik(ipred, iobs, sigiobs)


@pytest.mark.parametrize('form,names', [('photon', ['log_a', 'log_b', 'log_c']),
                                        ('gscaled', ['log_a', 'log_b', 'log_c']),
                                        ('aimless', ['sdfac', 'sdb', 'sdadd'])])
def test_the_form_parameters_train(form, names):
    ipred, iobs, sigiobs, s = _batch()
    lik = Ev11Likelihood(a=2., b=0.1, degrees_of_freedom=32., form=form)
    lik(ipred, iobs, sigiobs, scale=s).sum().backward()
    for name in names:
        g = getattr(lik, name).grad
        assert g is not None and torch.isfinite(g) and g != 0.
    trainable = {n for n, p in lik.named_parameters() if p.requires_grad}
    assert trainable == set(names)


def test_the_new_forms_refuse_the_online_fit():
    with pytest.raises(ValueError):
        Ev11Likelihood(form='aimless', fit='online')
