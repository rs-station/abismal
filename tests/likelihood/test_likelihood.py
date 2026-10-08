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
