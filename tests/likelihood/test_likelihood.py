import numpy as np
import pytest
import torch
from scipy.stats import norm, t as student_t
from abismal import tracking
from abismal.likelihood import NormalLikelihood, StudentTLikelihood, LaplaceLikelihood
from abismal.likelihood.location_scale import weighted_pearsonr


@pytest.mark.parametrize('df', [None, 32.])
def test_likelihood(df):
    if df is None:
        likelihood = NormalLikelihood()
    else:
        likelihood = StudentTLikelihood(df)

    mc_samples = 5
    refls = 10
    ipred = torch.rand((refls, mc_samples))
    iobs, sigiobs = torch.rand((2, refls, 1)).unbind(0)
    l = likelihood(ipred, iobs, sigiobs)
    assert l.shape == (refls, mc_samples)
    assert torch.isfinite(l).all()

    if df is None:
        expected = norm.logpdf(ipred.numpy(), iobs.numpy(), sigiobs.numpy())
    else:
        expected = student_t.logpdf(ipred.numpy(), df, iobs.numpy(), sigiobs.numpy())
    assert np.allclose(l.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_laplace_runs():
    l = LaplaceLikelihood()(torch.rand(4, 3), torch.rand(4, 1), torch.rand(4, 1) + 0.1)
    assert torch.isfinite(l).all()


def test_register_metrics():
    rng = np.random.default_rng(0)
    iobs = rng.normal(size=(100, 1)).astype('float32')
    sigiobs = rng.uniform(0.5, 1.5, size=(100, 1)).astype('float32')
    ipred = iobs + 0.1 * rng.normal(size=(100, 4)).astype('float32')
    with tracking.track() as tracker:
        NormalLikelihood().register_metrics(
            torch.as_tensor(ipred), torch.as_tensor(iobs), torch.as_tensor(sigiobs)
        )
    expected = np.corrcoef(np.broadcast_to(iobs, ipred.shape).ravel(), ipred.ravel())[0, 1]
    assert np.isclose(tracker.metrics['CCpred'].item(), expected, rtol=1e-4)
    assert 0.9 < tracker.metrics['wCCpred'].item() <= 1.0


def test_weighted_pearsonr_matches_numpy_with_unit_weights():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, 3, 50))
    r = weighted_pearsonr(torch.as_tensor(x), torch.as_tensor(y))
    expected = [np.corrcoef(a, b)[0, 1] for a, b in zip(x, y)]
    assert np.allclose(r.numpy(), expected)


@pytest.mark.parametrize('df', [None, 32.])
def test_ev11_likelihood(df):
    from abismal.likelihood import Ev11Likelihood
    likelihood = Ev11Likelihood(a=2., b=0.1, degrees_of_freedom=df)
    ipred = torch.rand((10, 5)) + 1.
    iobs, sigiobs = torch.rand((2, 10, 1)).unbind(0)
    l = likelihood(ipred, iobs, sigiobs)
    sd = np.sqrt(2. * sigiobs.numpy() ** 2 + (0.1 * ipred.numpy().mean(-1, keepdims=True)) ** 2)
    expected = norm.logpdf(ipred.numpy(), iobs.numpy(), sd) if df is None else student_t.logpdf(ipred.numpy(), df, iobs.numpy(), sd)
    assert np.allclose(l.detach().numpy(), expected, rtol=1e-5, atol=1e-5)

    # The variance's intensity is detached: the prediction gets gradient through the location alone.
    ipred.requires_grad_(True)
    likelihood(ipred, iobs, sigiobs).sum().backward()
    z = (ipred.detach() - iobs) / torch.from_numpy(sd).float()
    loc_grad = -z / torch.from_numpy(sd).float() if df is None else None
    if loc_grad is not None:
        assert torch.allclose(ipred.grad, loc_grad, rtol=1e-4, atol=1e-5)
    assert likelihood.log_a.grad is not None and likelihood.log_b.grad is not None


def test_ev11_recovers_a_from_a_posterior_with_the_right_width():
    # n observations of each reflection with noise variance a = 4 (σ = 1, s = 1, b ≈ 0). The posterior from n
    # observations under a flat prior is N(mean, a/n). The expected log likelihood over it, maximized in a, gives a
    # back: the posterior's own variance corrects for each observation's pull on the mean, with no leave-one-out.
    from abismal.likelihood import Ev11Likelihood
    torch.manual_seed(0)
    nref, n, a_true, mc = 2000, 4, 4., 64
    x = 10. + a_true ** 0.5 * torch.randn(nref, n)
    mu, sd = x.mean(-1, keepdim=True), (a_true / n) ** 0.5
    iobs = x.reshape(-1, 1)
    sigiobs = torch.ones_like(iobs)
    likelihood = Ev11Likelihood(a=1., b=1e-6)
    likelihood.log_b.requires_grad_(False)
    opt = torch.optim.Adam([likelihood.log_a], lr=0.05)
    for _ in range(300):
        z = (mu + sd * torch.randn(nref, mc)).repeat_interleave(n, 0)
        opt.zero_grad()
        (-likelihood(z, iobs, sigiobs).mean()).backward()
        opt.step()
    assert abs(float(likelihood.a) / a_true - 1.) < 0.05


def test_ev11_without_the_detach_passes_gradient_through_the_variance():
    from abismal.likelihood import Ev11Likelihood
    ipred = (torch.rand((10, 5)) + 1.).requires_grad_(True)
    iobs, sigiobs = torch.rand((2, 10, 1)).unbind(0)
    grads = []
    for detach in (True, False):
        ipred.grad = None
        Ev11Likelihood(b=0.5, detach=detach)(ipred, iobs, sigiobs).sum().backward()
        grads.append(ipred.grad.clone())
    assert not torch.allclose(grads[0], grads[1])


def test_a_fixed_ev11_model_has_nothing_to_train():
    from abismal.likelihood import Ev11Likelihood
    likelihood = Ev11Likelihood(a=0.994, b=0.047, fixed=True)
    assert not any(p.requires_grad for p in likelihood.parameters())
    assert abs(float(likelihood.a) - 0.994) < 1e-6 and abs(float(likelihood.b) - 0.047) < 1e-6
