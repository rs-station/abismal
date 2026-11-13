"""Tests for the lazy AdamW optimizer.

The central property under test is that a lazy parameter is decayed only on steps
where it actually received a gradient. Ordinary decoupled decay touches every
parameter on every step, so a sparsely-updated parameter would otherwise be pulled
toward zero at a rate set by how rarely it is observed.
"""
import numpy as np
import pytest
import torch

from abismal.optimizers import Adam, AdamW, AdaBelief


LR = 0.1
WD = 0.25


def _var(value):
    return torch.nn.Parameter(torch.tensor(np.asarray(value, dtype='float32')))


def _opt(cls, var, lazy=False, **kwargs):
    return cls([{'params': [var], 'lazy': lazy}], learning_rate=LR, **kwargs)


def _step(opt, var, grad):
    var.grad = torch.tensor(np.asarray(grad, dtype='float32'))
    opt.step()


def _np(var):
    return var.detach().numpy()


def test_default_weight_decay_matches_keras():
    """AdamW's default should match tf_keras.optimizers.AdamW rather than be 0."""
    assert _opt(AdamW, _var([1.])).weight_decay == 0.004
    # the other abismal optimizers keep decay off by default
    assert _opt(Adam, _var([1.])).weight_decay is None


def test_zero_decay_reduces_to_adam():
    """With weight_decay=0 the update must equal plain Adam exactly."""
    for lazy in (False, True):
        va, vw = _var([1.0, -2.0]), _var([1.0, -2.0])
        a = _opt(Adam, va, lazy)
        w = _opt(AdamW, vw, lazy, weight_decay=0.0)
        for g in ([0.5, 0.5], [0.1, -0.3], [0.2, 0.2]):
            _step(a, va, g)
            _step(w, vw, g)
        assert np.allclose(_np(va), _np(vw), rtol=1e-6)


def test_dense_variable_matches_manual_decoupled_decay():
    """A dense variable gets decoupled decay: v -= v*wd*lr, then Adam."""
    var = _var([1.0, -2.0])
    opt = _opt(AdamW, var, weight_decay=WD)
    ref_var = _var([1.0, -2.0])
    ref = _opt(Adam, ref_var)

    grad = [0.5, -0.25]
    _step(opt, var, grad)

    # replicate: decay applied to the pre-update value, then the Adam step
    with torch.no_grad():
        ref_var.mul_(1.0 - WD * LR)
    _step(ref, ref_var, grad)

    assert np.allclose(_np(var), _np(ref_var), rtol=1e-5)


def test_lazy_variable_with_zero_gradient_is_untouched():
    """The core property: no gradient, no decay, no change at all."""
    var = _var([1.0, 2.0, 3.0])
    opt = _opt(AdamW, var, lazy=True, weight_decay=WD)

    before = _np(var).copy()
    for _ in range(5):
        _step(opt, var, [0.0, 0.0, 0.0])
    assert np.array_equal(_np(var), before)


def test_lazy_decay_is_applied_only_to_updated_entries():
    """Within one variable, entries with zero gradient must not decay."""
    var = _var([1.0, 1.0])
    opt = _opt(AdamW, var, lazy=True, weight_decay=WD)

    # only the first entry has a gradient
    for _ in range(3):
        _step(opt, var, [0.5, 0.0])

    v = _np(var)
    assert v[1] == 1.0, "untouched entry must not be decayed"
    assert v[0] < 1.0, "updated entry should have moved"


def test_dense_variable_decays_even_without_gradient():
    """Contrast with the lazy case: this is the dense behavior lazy groups opt out of."""
    var = _var([1.0, 1.0])
    opt = _opt(AdamW, var, weight_decay=WD)

    _step(opt, var, [0.0, 0.0])
    assert np.all(_np(var) < 1.0)


def test_lazy_variable_decays_toward_zero_when_updated():
    """A lazy variable receiving gradients should still shrink under decay."""
    v0, v1 = _var([5.0]), _var([5.0])
    no_decay = _opt(AdamW, v0, lazy=True, weight_decay=0.0)
    decay = _opt(AdamW, v1, lazy=True, weight_decay=WD)

    for _ in range(10):
        _step(no_decay, v0, [0.01])
        _step(decay, v1, [0.01])

    assert _np(v1)[0] < _np(v0)[0]


def test_use_weight_decay_excludes_lazy_groups():
    """The dense decay must skip lazy groups, or they decay twice."""
    opt = AdamW([_var([1.0])], learning_rate=LR, weight_decay=WD)
    assert opt._use_weight_decay({'lazy': True}) is False
    assert opt._use_weight_decay({'lazy': False}) is True


def test_no_lazy_groups_behaves_like_torch_adamw():
    """With no lazy groups this is ordinary decoupled AdamW."""
    var = _var([1.0])
    opt = _opt(AdamW, var, weight_decay=WD, beta_1=0.9, beta_2=0.9, epsilon=1e-12)

    ref_var = _var([1.0])
    ref = torch.optim.AdamW([ref_var], lr=LR, weight_decay=WD, betas=(0.9, 0.9), eps=1e-12)
    for g in ([0.3], [0.1], [-0.2]):
        _step(opt, var, g)
        ref_var.grad = torch.tensor([g[0]])
        ref.step()
    # torch puts epsilon outside the bias-corrected root; keras inside. With
    # epsilon this small the two agree.
    assert np.allclose(_np(var), _np(ref_var), rtol=1e-4)


def test_weight_decay_none_is_safe():
    """weight_decay=None must not crash the lazy branch."""
    var = _var([1.0, 2.0])
    opt = _opt(AdamW, var, lazy=True, weight_decay=None)
    _step(opt, var, [0.5, 0.0])
    assert np.all(np.isfinite(_np(var)))
    assert _np(var)[1] == 2.0


def test_state_dict_round_trip():
    var = _var([1.0, 2.0])
    opt = _opt(AdamW, var, lazy=True, weight_decay=WD, beta_1=0.8, beta_2=0.95)
    _step(opt, var, [0.5, 0.1])
    # state_dict hands out live references to the moments; a saved file would not
    import copy
    state = copy.deepcopy(opt.state_dict())
    assert state['iterations'] == 1

    clone_var = _var(_np(var))
    clone = _opt(AdamW, clone_var, lazy=True, weight_decay=WD, beta_1=0.8, beta_2=0.95)
    clone.load_state_dict(state)
    assert clone.iterations == 1
    _step(opt, var, [0.2, 0.3])
    _step(clone, clone_var, [0.2, 0.3])
    assert np.array_equal(_np(var), _np(clone_var))


def test_registered_in_optimizer_dict():
    from abismal.optimizers.optimizer_dict import optimizer_dict

    assert optimizer_dict['adamw'] is AdamW
    assert optimizer_dict['adam'] is Adam
    assert optimizer_dict['adabelief'] is AdaBelief


@pytest.mark.parametrize('cls', [Adam, AdamW, AdaBelief])
def test_lazy_entries_untouched_for_every_optimizer(cls):
    var = _var([1.0, 2.0])
    opt = _opt(cls, var, lazy=True)
    for _ in range(3):
        _step(opt, var, [0.5, 0.0])
    assert np.all(np.isfinite(_np(var)))
    assert _np(var)[1] == 2.0


def test_learning_rate_schedule():
    from abismal.training import piecewise_constant
    var = _var([1.0])
    opt = AdamW([var], learning_rate=piecewise_constant(1, 0.1, 0.0), weight_decay=0.)
    _step(opt, var, [1.0])
    _step(opt, var, [1.0])
    moved = _np(var).copy()
    # Step 2 is past the boundary, where the rate is zero
    _step(opt, var, [1.0])
    assert np.array_equal(_np(var), moved)


def test_cosine_decay_runs_from_initial_to_final_and_holds():
    from abismal.training import cosine_decay
    lr = cosine_decay(100, 1e-3, 1e-5)
    assert lr(0) == pytest.approx(1e-3)
    assert lr(50) == pytest.approx(0.5 * (1e-3 + 1e-5))
    assert lr(100) == pytest.approx(1e-5) and lr(250) == pytest.approx(1e-5)
    assert all(lr(s) > lr(s + 1) for s in range(100))
    with pytest.raises(ValueError):
        cosine_decay(0, 1e-3, 1e-5)


def test_build_optimizer_picks_the_schedule():
    from types import SimpleNamespace
    from abismal.build import build_optimizer
    w = torch.nn.Parameter(torch.zeros(1))
    model = SimpleNamespace(parameter_groups=lambda: {"q": [], "scale": [w]})
    base = dict(learning_rate=1e-3, learning_rate_final=1e-4, burnin=10, epochs=2, steps_per_epoch=50,
                lr_schedule='cosine', beta_1=0.9, beta_2=0.999, adam_epsilon=1e-8, clipnorm=None, clip=None,
                global_clipnorm=None, optimizer='adam', weight_decay=None, per_reflection_bias_correction=False)
    build = lambda **kw: build_optimizer(SimpleNamespace(**{**base, **kw}), model)
    # cosine over epochs x steps per epoch = 100 steps, ignoring --burnin
    assert build().learning_rate(0) == pytest.approx(1e-3)
    assert build().learning_rate(50) == pytest.approx(0.5 * (1e-3 + 1e-4))
    assert build().learning_rate(100) == pytest.approx(1e-4)
    step = build(lr_schedule='step').learning_rate
    assert step(10) == pytest.approx(1e-3) and step(11) == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        build(learning_rate_final=None)
    # Without --burnin the step drops two thirds of the way through training
    third = build(lr_schedule='step', burnin=None).learning_rate
    assert third(66) == pytest.approx(1e-3) and third(67) == pytest.approx(1e-4)
    # ... unless the training length is unknown (one pass per epoch), when it holds
    assert build(lr_schedule='step', burnin=None, steps_per_epoch=0).learning_rate == pytest.approx(1e-3)


def test_the_default_drops_to_1e4_at_step_20000():
    from abismal.command_line.parser import parser
    from abismal.build import build_optimizer
    from types import SimpleNamespace
    args = SimpleNamespace(**{a.dest: a.default for a in parser._actions if a.dest != "help"})
    assert args.learning_rate_final == 1e-4 and args.burnin is None and args.lr_schedule == 'step'
    w = torch.nn.Parameter(torch.zeros(1))
    model = SimpleNamespace(parameter_groups=lambda: {"q": [], "scale": [w]})
    lr = build_optimizer(args, model).learning_rate
    assert lr(20000) == pytest.approx(1e-3) and lr(20001) == pytest.approx(1e-4)


@pytest.mark.parametrize('kwargs', [dict(clipvalue=0.1), dict(clipnorm=0.1), dict(global_clipnorm=0.1)])
def test_clipping_bounds_the_first_step(kwargs):
    """Adam's first step is lr * sign(g) whatever the scale, so compare moments instead."""
    var = _var([1.0, 1.0])
    opt = _opt(Adam, var, **kwargs)
    _step(opt, var, [10.0, -10.0])
    m = opt.state[var]['m']
    # m = (1 - beta_1) * clipped gradient
    clipped = m / (1 - opt.beta_1)
    if 'clipvalue' in kwargs:
        assert torch.allclose(clipped.abs(), torch.tensor(0.1))
    else:
        assert torch.isclose(torch.linalg.vector_norm(clipped), torch.tensor(0.1))


@pytest.mark.parametrize("cls", [Adam, AdaBelief])
def test_per_entry_bias_correction_leaves_dense_groups_and_the_default_alone(cls):
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(20, 3)).astype('float32')
    out = []
    for kwargs in ({}, {'per_entry_bias_correction': True}):
        var = _var([0., 0., 0.])
        opt = _opt(cls, var, lazy=False, **kwargs)
        for g in grads:
            _step(opt, var, g)
        out.append(_np(var))
    np.testing.assert_array_equal(out[0], out[1])


def test_per_entry_bias_correction_makes_each_lazy_entry_follow_its_own_adam():
    """Each entry should move exactly as torch's Adam would on that entry's own nonzero gradients."""
    rng = np.random.default_rng(1)
    grads = rng.normal(size=(200, 3)).astype('float32')
    grads[rng.random(grads.shape) > np.array([1., 0.3, 0.02])] = 0.   # entry 2 is updated on ~2% of steps
    var = _var([0., 0., 0.])
    opt = _opt(Adam, var, lazy=True, per_entry_bias_correction=True, beta_1=0.9, beta_2=0.999, epsilon=1e-12)
    for g in grads:
        _step(opt, var, g)
    for i in range(3):
        ref = torch.nn.Parameter(torch.zeros(1))
        torch_adam = torch.optim.Adam([ref], lr=LR, betas=(0.9, 0.999), eps=1e-12)
        for g in grads[:, i][grads[:, i] != 0.]:
            ref.grad = torch.tensor([g])
            torch_adam.step()
        np.testing.assert_allclose(_np(var)[i], ref.item(), rtol=1e-4, atol=1e-6)


def test_without_per_entry_bias_correction_a_rare_entry_overshoots_adam():
    """The global count under-corrects a rarely updated entry: its early steps are larger than Adam's."""
    var = _var([0., 0.])
    opt = _opt(Adam, var, lazy=True, beta_1=0.9, beta_2=0.999, epsilon=1e-12)
    for t in range(500):
        _step(opt, var, [1., 1. if t >= 490 else 0.])  # entry 1 is updated on the last 10 steps only
    # Adam's steps are ~LR each for a constant gradient, so 10 of them move it ~10 LR; the global count gives far more
    assert abs(_np(var)[1]) > 3 * 10 * LR
