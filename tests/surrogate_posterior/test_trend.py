import numpy as np
import pytest
import torch
from abismal import tracking
from abismal.surrogate_posterior.trend import ResolutionTrend
from model_helpers import build_small_model, synthetic_batch


def _with_trend(kind, **kwargs):
    model = build_small_model(**kwargs)
    model.surrogate_posterior.trend = ResolutionTrend(model.surrogate_posterior.rac, kind, knots=4)
    return model


def test_a_wilson_trend_is_k_exp_minus_b_over_d_squared():
    rac = build_small_model().surrogate_posterior.rac
    t = ResolutionTrend(rac, 'wilson')
    assert torch.equal(t.log_trend(), torch.zeros_like(t.log_trend()))
    with torch.no_grad():
        t._log_k.fill_(.1); t._b.fill_(2.)
    q = torch.as_tensor(rac.dHKL).reshape(-1) ** -2
    assert torch.allclose(t.log_trend(), 1. - 20. * q, rtol=1e-4)


def test_a_spline_trend_interpolates_its_knots():
    rac = build_small_model().surrogate_posterior.rac
    t = ResolutionTrend(rac, 'spline', knots=4)
    with torch.no_grad():
        t._values.copy_(.1 * t.knots * 3. + .05)  # log T linear in 1/d²: the interpolation must reproduce it
    q = torch.as_tensor(rac.dHKL, dtype=torch.float32).reshape(-1) ** -2
    assert torch.allclose(t.log_trend(), 3. * q + .5, atol=1e-5)


@pytest.mark.parametrize('kind', ['wilson', 'spline'])
def test_the_trend_trains_with_every_step_and_not_lazily(kind):
    from abismal.training import parameter_groups
    model = _with_trend(kind)
    trend = {id(p) for p in model.surrogate_posterior.trend.parameters()}
    groups = parameter_groups(model)
    lazy = {id(p) for g in groups if g['lazy'] for p in g['params']}
    dense = {id(p) for g in groups if not g['lazy'] for p in g['params']}
    assert trend <= dense and not (trend & lazy)
    batch = synthetic_batch(model).map(torch.as_tensor)
    with tracking.track() as t:
        model(batch, training=True)
    t.total_loss().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.surrogate_posterior.trend.parameters())


def test_the_trend_multiplies_the_prediction():
    model = _with_trend('wilson')
    batch = synthetic_batch(model).map(torch.as_tensor)
    with tracking.track():
        model(batch, training=True)  # the standardization's first moments
    model.eval()
    def predict():
        torch.manual_seed(0)
        with tracking.track(), torch.no_grad():
            return model(batch, training=False)
    flat = predict()
    with torch.no_grad():
        model.surrogate_posterior.trend._log_k.fill_(np.log(4.) / 10.)
    assert torch.allclose(predict(), 4. * flat, rtol=1e-4)


def test_the_trend_multiplies_the_output():
    model = _with_trend('wilson')
    sp = model.surrogate_posterior
    sp.seen.fill_(True)
    before = next(iter(_flat(sp, None)))
    with torch.no_grad():
        sp.trend._log_k.fill_(np.log(4.) / 10.)
    F, I, SIGF = before
    after = next(iter(_flat(sp, sp.trend)))
    assert np.allclose(after[0], 2. * F, rtol=1e-4)  # F and SIGF by sqrt(T)
    assert np.allclose(after[1], 4. * I, rtol=1e-4)  # I by T
    assert np.allclose(after[2], 2. * SIGF, rtol=1e-4)


def _flat(sp, trend):
    saved, sp.trend = sp.trend, trend
    try:
        for ds in sp.to_datasets():
            ds = ds.stack_anomalous()
            yield ds['F'].to_numpy(float), ds['I'].to_numpy(float), ds['SIGF'].to_numpy(float)
    finally:
        sp.trend = saved
