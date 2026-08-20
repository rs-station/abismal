"""Tests for the lazy AdamW optimizer.

The central property under test is that a lazy variable is decayed only on steps
where it actually received a gradient. The base Keras Optimizer decays every
variable on every step, so a sparsely-updated parameter would otherwise be pulled
toward zero at a rate set by how rarely it is observed.
"""
import numpy as np
import pytest
import tensorflow as tf
import tf_keras as tfk

from abismal.optimizers import Adam, AdamW


LR = 0.1
WD = 0.25


def _var(value, name='v'):
    return tf.Variable(np.asarray(value, dtype='float32'), name=name)


def _step(opt, var, grad):
    opt.apply_gradients([(tf.constant(np.asarray(grad, dtype='float32')), var)])


def test_default_weight_decay_matches_keras():
    """AdamW's default should match tf_keras.optimizers.AdamW rather than be 0."""
    assert AdamW().weight_decay == 0.004
    assert AdamW().weight_decay == tfk.optimizers.AdamW().weight_decay
    # the other abismal optimizers keep decay off by default
    assert Adam().weight_decay is None


def test_zero_decay_reduces_to_adam():
    """With weight_decay=0 the update must equal plain Adam exactly."""
    for lazy in (None, ['x']):
        a = Adam(learning_rate=LR, lazy_vars=lazy)
        w = AdamW(learning_rate=LR, weight_decay=0.0, lazy_vars=lazy)
        va, vw = _var([1.0, -2.0]), _var([1.0, -2.0])
        if lazy is not None:
            # lazy_vars is matched on _var_key, so use the real ids
            a.lazy_vars = [a._var_key(va)]
            w.lazy_vars = [w._var_key(vw)]
        for g in ([0.5, 0.5], [0.1, -0.3], [0.2, 0.2]):
            _step(a, va, g)
            _step(w, vw, g)
        assert np.allclose(va.numpy(), vw.numpy(), rtol=1e-6)


def test_dense_variable_matches_manual_decoupled_decay():
    """A dense variable gets base-class decoupled decay: v -= v*wd*lr, then Adam."""
    opt = AdamW(learning_rate=LR, weight_decay=WD)
    var = _var([1.0, -2.0])
    ref = Adam(learning_rate=LR)
    ref_var = _var([1.0, -2.0])

    grad = [0.5, -0.25]
    _step(opt, var, grad)

    # replicate: decay applied to the pre-update value, then the Adam step
    decayed = ref_var.numpy() * (1.0 - WD * LR)
    ref_var.assign(decayed)
    _step(ref, ref_var, grad)

    assert np.allclose(var.numpy(), ref_var.numpy(), rtol=1e-5)


def test_lazy_variable_with_zero_gradient_is_untouched():
    """The core property: no gradient, no decay, no change at all."""
    opt = AdamW(learning_rate=LR, weight_decay=WD)
    var = _var([1.0, 2.0, 3.0])
    opt.lazy_vars = [opt._var_key(var)]

    before = var.numpy().copy()
    for _ in range(5):
        _step(opt, var, [0.0, 0.0, 0.0])
    assert np.array_equal(var.numpy(), before)


def test_lazy_decay_is_applied_only_to_updated_entries():
    """Within one variable, entries with zero gradient must not decay."""
    opt = AdamW(learning_rate=LR, weight_decay=WD)
    var = _var([1.0, 1.0])
    opt.lazy_vars = [opt._var_key(var)]

    # only the first entry has a gradient
    for _ in range(3):
        _step(opt, var, [0.5, 0.0])

    v = var.numpy()
    assert v[1] == 1.0, "untouched entry must not be decayed"
    assert v[0] < 1.0, "updated entry should have moved"


def test_dense_variable_decays_even_without_gradient():
    """Contrast with the lazy case: this is the base-class behavior we opt out of."""
    opt = AdamW(learning_rate=LR, weight_decay=WD)
    var = _var([1.0, 1.0])  # not registered as lazy

    _step(opt, var, [0.0, 0.0])
    assert np.all(var.numpy() < 1.0)


def test_lazy_variable_decays_toward_zero_when_updated():
    """A lazy variable receiving gradients should still shrink under decay."""
    no_decay = AdamW(learning_rate=LR, weight_decay=0.0)
    decay = AdamW(learning_rate=LR, weight_decay=WD)
    v0, v1 = _var([5.0]), _var([5.0])
    no_decay.lazy_vars = [no_decay._var_key(v0)]
    decay.lazy_vars = [decay._var_key(v1)]

    for _ in range(10):
        _step(no_decay, v0, [0.01])
        _step(decay, v1, [0.01])

    assert v1.numpy()[0] < v0.numpy()[0]


def test_use_weight_decay_excludes_lazy_vars():
    """The base class must be told to skip lazy vars, or they decay twice."""
    opt = AdamW(learning_rate=LR, weight_decay=WD)
    lazy, dense = _var([1.0], name='lazy'), _var([1.0], name='dense')
    opt.lazy_vars = [opt._var_key(lazy)]

    assert opt._use_weight_decay(lazy) is False
    assert opt._use_weight_decay(dense) is True


def test_no_lazy_vars_behaves_like_plain_adamw():
    """lazy_vars=None must not trip the exclusion logic."""
    opt = AdamW(learning_rate=LR, weight_decay=WD, lazy_vars=None)
    var = _var([1.0])
    assert opt._use_weight_decay(var) is True

    ref = tfk.optimizers.AdamW(
        learning_rate=LR, weight_decay=WD, beta_1=0.9, beta_2=0.9, epsilon=1e-12,
    )
    ref_var = _var([1.0])
    for g in ([0.3], [0.1], [-0.2]):
        _step(opt, var, g)
        _step(ref, ref_var, g)
    assert np.allclose(var.numpy(), ref_var.numpy(), rtol=1e-4)


def test_weight_decay_none_is_safe():
    """weight_decay=None must not crash the lazy branch."""
    opt = AdamW(learning_rate=LR, weight_decay=None)
    var = _var([1.0, 2.0])
    opt.lazy_vars = [opt._var_key(var)]
    _step(opt, var, [0.5, 0.0])
    assert np.all(np.isfinite(var.numpy()))
    assert var.numpy()[1] == 2.0


def test_serialization_round_trip():
    opt = AdamW(learning_rate=LR, weight_decay=WD, beta_1=0.8, beta_2=0.95)
    config = opt.get_config()
    assert config['weight_decay'] == WD

    restored = AdamW.from_config(config)
    assert restored.weight_decay == WD
    assert restored.beta_1 == 0.8
    assert restored.beta_2 == 0.95
    assert isinstance(restored, AdamW)


def test_registered_in_optimizer_dict():
    from abismal.optimizers.optimizer_dict import optimizer_dict

    assert optimizer_dict['adamw'] is AdamW
    assert 'tfkadamw' in optimizer_dict


@pytest.mark.parametrize('jit_compile', [False, True])
def test_runs_under_jit(jit_compile):
    """update_step must work as a compiled op, not just eagerly."""
    opt = AdamW(learning_rate=LR, weight_decay=WD, jit_compile=jit_compile)
    var = _var([1.0, 2.0])
    opt.lazy_vars = [opt._var_key(var)]
    for _ in range(3):
        _step(opt, var, [0.5, 0.0])
    assert np.all(np.isfinite(var.numpy()))
    assert var.numpy()[1] == 2.0
