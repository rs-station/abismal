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


_CELLS = {
    'P 1': (40., 50., 60., 80., 95., 105.),
    'P 1 2 1': (40., 50., 60., 90., 105., 90.),
    'P 21 21 21': (40., 50., 60., 90., 90., 90.),
    'P 43 21 2': (79., 79., 38., 90., 90., 90.),
    'H 3': (90., 90., 60., 90., 90., 120.),
    'P 61 2 2': (93., 93., 131., 90., 90., 120.),
    'P 2 3': (50., 50., 50., 90., 90., 90.),
}
_ALLOWED = {'P 1': 5, 'P 1 2 1': 3, 'P 21 21 21': 2, 'P 43 21 2': 1, 'H 3': 1, 'P 61 2 2': 1, 'P 2 3': 0}


def _rac(spacegroup):
    from abismal.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUCollection
    import gemmi
    return ReciprocalASUCollection(ReciprocalASU(gemmi.UnitCell(*_CELLS[spacegroup]), gemmi.SpaceGroup(spacegroup), 4.))


@pytest.mark.parametrize('spacegroup', list(_CELLS))
def test_the_anisotropic_b_has_the_freedom_the_laue_symmetry_allows(spacegroup):
    t = ResolutionTrend(_rac(spacegroup), 'spline', knots=4, aniso='global')
    assert np.linalg.matrix_rank(t.projection[0].numpy(), tol=1e-5) == _ALLOWED[spacegroup]


@pytest.mark.parametrize('spacegroup', list(_CELLS))
def test_symmetry_mates_get_the_same_anisotropic_falloff(spacegroup):
    import gemmi
    rac = _rac(spacegroup)
    t = ResolutionTrend(rac, 'spline', knots=4, aniso='bins')
    with torch.no_grad():
        t._aniso.normal_()
    b = t.b_tensors()[0].detach().numpy().astype(float)
    assert np.allclose(b[:, :3].sum(-1), 0., atol=1e-4)  # traceless: the isotropic part is the spline's
    cell, ops = gemmi.UnitCell(*_CELLS[spacegroup]), gemmi.SpaceGroup(spacegroup).operations()
    frac = np.array(cell.frac.mat.tolist())
    for h in ([1, 2, 3], [4, -1, 2], [0, 3, 5]):
        s = np.array([op.apply_to_hkl(h) for op in ops.sym_ops], dtype=float) @ frac
        sss = np.stack([s[:, 0] ** 2, s[:, 1] ** 2, s[:, 2] ** 2, 2 * s[:, 0] * s[:, 1], 2 * s[:, 0] * s[:, 2], 2 * s[:, 1] * s[:, 2]], -1)
        quad = sss @ b.T  # each mate against every knot's B
        assert np.allclose(quad, quad[:1], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('aniso', ['global', 'bins'])
def test_an_anisotropic_trend_starts_isotropic_and_trains(aniso):
    model = build_small_model()
    rac = model.surrogate_posterior.rac
    model.surrogate_posterior.trend = ResolutionTrend(rac, 'spline', knots=4, aniso=aniso)
    assert torch.equal(model.surrogate_posterior.trend.log_trend(), ResolutionTrend(rac, 'spline', knots=4).log_trend())
    batch = synthetic_batch(model).map(torch.as_tensor)
    with tracking.track() as t:
        model(batch, training=True)
    t.total_loss().backward()
    assert model.surrogate_posterior.trend._aniso.grad.abs().sum() > 0


def test_an_anisotropic_trend_needs_a_spline():
    with pytest.raises(ValueError):
        ResolutionTrend(build_small_model().surrogate_posterior.rac, 'wilson', aniso='global')
