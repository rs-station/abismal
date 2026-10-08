"""
Regression tests for the freezing policy abismal.cchalf applies to the model.
"""
import torch
import pytest

from abismal.command_line.cchalf import freeze_all_but_posterior
from abismal.optimizers import Adam
from abismal.training import Trainer, parameter_groups
from conftest import build_small_model, synthetic_batch


@pytest.fixture
def model():
    return build_small_model()


def test_freeze_leaves_only_the_posterior_trainable(model):
    freeze_all_but_posterior(model)

    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    posterior = {id(p) for p in model.surrogate_posterior.parameters()}
    assert len(posterior) > 0, "the posterior itself reports nothing to train"
    assert trainable == posterior


def test_parameter_groups_after_freezing(model):
    """The optimizer is built from parameter_groups, so it must see only the posterior, lazily."""
    freeze_all_but_posterior(model)
    groups = parameter_groups(model)
    assert len(groups) == 1
    assert groups[0]['lazy'] is True
    assert {id(p) for p in groups[0]['params']} == {id(p) for p in model.surrogate_posterior.parameters()}


def test_freeze_recovers_a_model_frozen_from_the_top(model):
    """cchalf reloads the checkpoint per half, so freezing must not be one-way."""
    model.requires_grad_(False)
    freeze_all_but_posterior(model)
    assert any(p.requires_grad for p in model.parameters())


def test_frozen_components_receive_no_updates(model):
    """The gradients really do stop at the frozen components, not just the listing."""
    freeze_all_but_posterior(model)
    trainer = Trainer(model, Adam(parameter_groups(model), learning_rate=0.1), verbose=0)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer.train_step(synthetic_batch(model).map(torch.as_tensor))
    after = model.state_dict()

    moved = {k for k in before if not torch.equal(before[k], after[k])}
    # The posterior's parameters move; so do buffers that track what was seen
    # and the running moments of the standardization layers, which no optimizer
    # touches. Nothing in the scale model, prior or likelihood may.
    for k in moved:
        assert k.startswith("surrogate_posterior.") or k.startswith("standardize_"), k
    assert any(k.startswith("surrogate_posterior.") and k.endswith(".raw") for k in moved)
