"""
Regression tests for the freezing policy abismal.cchalf applies to the model.
"""
import numpy as np
import tf_keras as tfk
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pytest

from abismal.command_line.cchalf import freeze_all_but_posterior


class Component(tfk.layers.Layer):
    def build(self, shape):
        self.w = self.add_weight(shape=(2,), initializer='ones', name='w')

    def call(self, x):
        return x * self.w[0]


class FakeMergingModel(tfk.models.Model):
    """
    Stands in for VariationalMergingModel: the attribute layout freeze_all_but_posterior
    touches, and nothing else.
    """
    def __init__(self):
        super().__init__()
        self.scale_model = Component()
        self.surrogate_posterior = Component()
        self.prior = Component()
        self.likelihood = Component()

    def call(self, x):
        return self.scale_model(x) + self.surrogate_posterior(x)


@pytest.fixture
def model():
    m = FakeMergingModel()
    m(np.ones((1, 2), dtype='float32'))
    for child in (m.prior, m.likelihood):
        child.build(None)
    return m


def test_freeze_leaves_posterior_trainable_from_the_top(model):
    """
    The posterior's variables must survive all the way up to model.trainable_variables.

    This is the assertion that would have caught the original bug. cchalf used to
    freeze by setting `model.trainable = False` and re-enabling the child, but
    tf_keras' `Layer.trainable_weights` short-circuits to `[]` when the layer itself
    is not trainable, no matter what its children say. Checking only
    `surrogate_posterior.trainable_variables` looks fine in that state -- the
    breakage is visible only from the model.
    """
    freeze_all_but_posterior(model)

    posterior_refs = {v.ref() for v in model.surrogate_posterior.trainable_variables}
    assert len(posterior_refs) > 0, "the posterior itself reports nothing to train"

    model_refs = {v.ref() for v in model.trainable_variables}
    assert model_refs == posterior_refs, (
        f"model.trainable_variables holds {len(model_refs)} variables but the "
        f"posterior holds {len(posterior_refs)}; cchalf optimizes what the *model* "
        f"exposes, so anything missing here is never updated"
    )


def test_freeze_excludes_every_other_component(model):
    freeze_all_but_posterior(model)

    trainable = {v.ref() for v in model.trainable_variables}
    for name in ('scale_model', 'prior', 'likelihood'):
        component = getattr(model, name)
        assert not component.trainable, f"{name} should be frozen"
        assert not trainable.intersection(v.ref() for v in component.weights), (
            f"{name} weights leaked into model.trainable_variables"
        )


def test_freeze_recovers_a_model_frozen_from_the_top(model):
    """cchalf reloads the checkpoint per half, so freezing must not be one-way."""
    model.trainable = False
    assert len(model.trainable_variables) == 0

    freeze_all_but_posterior(model)
    assert len(model.trainable_variables) > 0, (
        "freeze_all_but_posterior did not re-enable the top-level model"
    )


def test_frozen_components_receive_no_updates(model):
    """The gradients really do stop at the frozen components, not just the listing."""
    freeze_all_but_posterior(model)
    x = np.ones((4, 2), dtype='float32')

    before = {v.ref(): v.numpy().copy() for v in model.weights}
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(model(x) - 3.0))
    variables = model.trainable_variables
    tfk.optimizers.SGD(learning_rate=0.1).apply_gradients(
        zip(tape.gradient(loss, variables), variables)
    )

    moved = {v.ref() for v in model.weights
             if not np.array_equal(v.numpy(), before[v.ref()])}
    expected = {v.ref() for v in model.surrogate_posterior.trainable_variables}
    assert moved == expected, (
        f"{len(moved)} weights moved under one SGD step, expected the "
        f"{len(expected)} posterior weights and nothing else"
    )
