"""WeightSaver writes a model that reloads to the same function.

epoch_0 is written before the first gradient step. Its consumer is
`abismal.cchalf --sf-init`, which wants the structure factor initialization, so
the file has to hold the whole model as it was constructed.
"""
import numpy as np
import torch

from abismal.callbacks import WeightSaver
from abismal.optimizers import Adam
from abismal.saving import load_model, load_optimizer
from abismal.training import parameter_groups
from conftest import build_small_model, synthetic_batch


class _Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = Adam(parameter_groups(model))


def _predict(model, batch):
    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        return model(batch.map(torch.as_tensor), training=False)


def test_epoch_zero_is_written_and_reloads_to_the_same_model(tmp_path):
    model = build_small_model()
    saver = WeightSaver(str(tmp_path))
    saver.set_trainer(_Trainer(model))
    saver.on_train_begin()

    path = tmp_path / "epoch_0.pt"
    assert path.exists()
    reloaded = load_model(str(path))
    assert type(reloaded) is type(model)

    for (k, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), k

    batch = synthetic_batch(model)
    assert torch.allclose(_predict(model, batch), _predict(reloaded, batch))


def test_epoch_end_saves_numbered_by_completed_epochs(tmp_path):
    model = build_small_model()
    saver = WeightSaver(str(tmp_path))
    saver.set_trainer(_Trainer(model))
    saver.on_epoch_end(0)
    saver.on_epoch_end(1)
    assert (tmp_path / "epoch_1.pt").exists()
    assert (tmp_path / "epoch_2.pt").exists()


def test_the_optimizer_is_saved_with_its_hyperparameters(tmp_path):
    model = build_small_model()
    trainer = _Trainer(model)
    trainer.optimizer.beta_2 = 0.95
    saver = WeightSaver(str(tmp_path))
    saver.set_trainer(trainer)
    saver.on_epoch_end(0)

    reloaded = load_model(str(tmp_path / "epoch_1.pt"))
    opt = load_optimizer(str(tmp_path / "epoch_1.pt"), reloaded)
    assert isinstance(opt, Adam)
    assert opt.beta_2 == 0.95
    assert [g['lazy'] for g in opt.param_groups] == [False, True]
