"""Tests for abismal.training, the loop that replaced keras' Model.fit."""
import re
import numpy as np
import pytest
import torch

from abismal.io.pipeline import ImageCache, BatchStream
from abismal.optimizers import Adam
from abismal.training import Trainer, EarlyStopping, parameter_groups, grad_rms
from abismal.callbacks.base import Callback
from model_helpers import build_small_model, synthetic_batch


def _images(model, n, seed=0):
    batch = synthetic_batch(model, num_images=n, refls_per_image=15, seed=seed)
    counts = np.bincount(batch.image_id, minlength=n)
    splits = np.cumsum(counts)[:-1]
    fields = [np.split(f, splits) for f in batch.inputs]
    return list(zip(*fields))


def _streams(model, n=30, test_fraction=0.3, repeat=False, batch_size=4):
    cache = ImageCache(_images(model, n), test_fraction=test_fraction)
    train = BatchStream(cache, 'train', batch_size=batch_size, shuffle_buffer_size=8, repeat=repeat, seed=0)
    test = BatchStream(cache, 'test', batch_size=batch_size, repeat=False)
    return train, test


class Recorder(Callback):
    def __init__(self):
        self.calls = []

    def on_train_begin(self, logs=None):
        self.calls.append('train_begin')

    def on_epoch_begin(self, epoch, logs=None):
        self.calls.append(f'begin {epoch}')

    def on_epoch_end(self, epoch, logs=None):
        self.calls.append(f'end {epoch}')
        self.last_logs = logs

    def on_train_end(self, logs=None):
        self.calls.append('train_end')


def test_fit_logs_training_and_validation_metrics(capsys):
    torch.manual_seed(0)
    model = build_small_model()
    train, test = _streams(model)
    recorder = Recorder()
    trainer = Trainer(model, Adam(parameter_groups(model), learning_rate=1e-2), callbacks=[recorder], verbose=2)
    history = trainer.fit(train, epochs=3, validation_data=test)

    assert recorder.calls == ['train_begin', 'begin 0', 'end 0', 'begin 1', 'end 1', 'begin 2', 'end 2', 'train_end']
    for key in ('loss', 'NLL', 'KL', 'CCpred', '|∇s|', '|∇q|', 'val_loss', 'val_NLL', 'val_CCpred'):
        assert key in history, key
        assert len(history[key]) == 3
        assert all(np.isfinite(history[key]))
    # Gradient norms are a training quantity
    assert 'val_|∇s|' not in history

    # The GUI's progress bar parses these lines
    out = capsys.readouterr().out
    assert re.findall(r'^Epoch (\d+)/(\d+)$', out, re.M) == [('1', '3'), ('2', '3'), ('3', '3')]


def test_training_reduces_the_loss():
    torch.manual_seed(0)
    model = build_small_model()
    train, _ = _streams(model, test_fraction=0.)
    trainer = Trainer(model, Adam(parameter_groups(model), learning_rate=1e-2), verbose=0)
    history = trainer.fit(train, epochs=15)
    assert history['loss'][-1] < history['loss'][0]


def test_repeating_stream_keeps_one_iterator_and_counts_steps():
    model = build_small_model()
    train, _ = _streams(model, test_fraction=0., repeat=True)
    trainer = Trainer(model, Adam(parameter_groups(model)), verbose=0)
    with pytest.raises(ValueError, match="steps_per_epoch"):
        trainer.fit(train, epochs=1)

    steps = []
    original = trainer.train_step
    def counting(batch):
        steps.append(batch.num_images)
        return original(batch)
    trainer.train_step = counting
    trainer.fit(train, epochs=2, steps_per_epoch=7)
    assert len(steps) == 14


def test_early_stopping_stops():
    model = build_small_model()
    train, test = _streams(model)
    stopper = EarlyStopping(monitor='loss', patience=0, mode='max')
    trainer = Trainer(model, Adam(parameter_groups(model), learning_rate=1e-2), callbacks=[stopper], verbose=0)
    history = trainer.fit(train, epochs=10)
    # A falling loss never "improves" in max mode, so it stops after epoch 2
    assert len(history['loss']) == 2


def test_nonfinite_gradients_are_zeroed():
    model = build_small_model()
    trainer = Trainer(model, Adam(parameter_groups(model)), verbose=0)
    batch = synthetic_batch(model).map(torch.as_tensor)
    # Poison one image's observations so its gradient contributions are nan
    iobs = batch.iobs.clone()
    iobs[0] = float('nan')
    before = [p.detach().clone() for p in model.parameters()]
    trainer.train_step(batch._replace(iobs=iobs))
    for p, b in zip(model.parameters(), before):
        assert torch.isfinite(p).all()


def test_grad_rms_of_an_empty_group_is_zero():
    assert grad_rms([]).item() == 0.


def test_bfloat16_scale_model_trains_and_reports_float32():
    torch.manual_seed(0)
    model = build_small_model()
    model.scale_model.precision = "bfloat16"
    trainer = Trainer(model, Adam(parameter_groups(model)), verbose=0)
    batch = synthetic_batch(model).map(torch.as_tensor)
    metrics = trainer.train_step(batch)
    assert all(torch.isfinite(v) for v in metrics.values())
    z = model.scale_model(batch, mc_samples=3, training=False)
    assert z.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiles for CUDA")
def test_compiled_step_matches_eager_loss():
    """The compiled loss is the eager loss, up to kernel-level rounding."""
    import torch._inductor.config as inductor_config
    # Inductor otherwise draws its own random numbers, which are not eager's
    # even from the same seed, so the Monte Carlo samples would differ.
    inductor_config.fallback_random = True
    model = build_small_model().cuda()
    batch = synthetic_batch(model).map(lambda a: torch.as_tensor(a).cuda())
    eager = Trainer(model, Adam(parameter_groups(model)), verbose=0)
    compiled = Trainer(model, Adam(parameter_groups(model)), verbose=0, compile={"dynamic": True})
    # Give the standardization layers their moments; a fresh model has none
    with torch.no_grad():
        eager._train_loss(batch)
    model.eval()
    torch.manual_seed(0)
    a, _ = eager._test_loss(batch)
    torch.manual_seed(0)
    b, _ = compiled._test_loss(batch)
    assert torch.allclose(a, b, rtol=1e-3)


def test_epoch_metrics_weight_batches_by_images():
    from abismal.training import _RunningMeans
    means = _RunningMeans()
    means.update({"NLL": torch.tensor(10.)}, weight=100)
    means.update({"NLL": torch.tensor(20.)}, weight=2)
    # Unweighted this would be 15; by images it is barely moved from 10
    assert abs(means.result()["NLL"] - (10. * 100 + 20. * 2) / 102) < 1e-5


def test_the_trainer_says_when_it_starts_training(capsys):
    """Uncompiled, it says Training at once; the GUI shows these [status] lines."""
    model = build_small_model()
    train, test = _streams(model)
    Trainer(model, Adam(parameter_groups(model)), verbose=0).fit(train, epochs=1, validation_data=test)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[status] ")]
    assert lines == ["[status] Training"]
