"""Holding out a fraction of every image's reflections for validation."""
import numpy as np
import pytest
import torch

from abismal.io.pipeline import ImageCache, BatchStream, holdout_mask
from abismal import tracking
from model_helpers import build_small_model, synthetic_batch


def _images(n=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(10, 60))
        out.append((
            np.zeros((k, 1), np.int32), rng.integers(-20, 20, (k, 3)).astype(np.int32),
            np.ones((k, 1), np.float32), np.ones((k, 1), np.float32),
            rng.normal(size=(k, 2)).astype(np.float32),
            rng.normal(size=(k, 1)).astype(np.float32), np.ones((k, 1), np.float32),
        ))
    return out


def test_mask_is_stable_and_about_the_right_size():
    images = _images(50)
    masks = [holdout_mask(im, 0.2) for im in images]
    again = [holdout_mask(im, 0.2) for im in images]
    assert all((a == b).all() for a, b in zip(masks, again))
    frac = np.concatenate(masks).mean()
    assert 0.15 < frac < 0.25
    # every image keeps a training reflection, even when almost all are held out
    assert all((~holdout_mask(im, 0.999)).any() for im in images)


def test_train_and_validate_partition_the_reflections():
    images = _images()
    cache = ImageCache(images)
    train = list(BatchStream(cache, batch_size=100, holdout_fraction=0.3, holdout_role='train'))
    val = list(BatchStream(cache, batch_size=100, holdout_fraction=0.3, holdout_role='validate'))
    total = sum(len(im[0]) for im in images)
    n_train = sum(len(b.iobs) for b in train)
    held = sum(int(b.loss_weight.sum()) for b in val)
    assert sum(len(b.iobs) for b in val) == total
    assert n_train + held == total
    assert all(((b.loss_weight + b.context_weight) == 1).all() for b in val)
    assert train[0].loss_weight is None and train[0].context_weight is None


def test_held_out_intensities_do_not_reach_the_training_reflections():
    """Changing a held-out intensity must not change any training reflection's scale."""
    torch.manual_seed(0)
    model = build_small_model(standardize=False)
    model.scale_model.metadata_noise_factor = 0.
    model.eval()
    batch = synthetic_batch(model).map(torch.as_tensor)
    holdout = torch.zeros(len(batch.iobs), dtype=torch.bool)
    holdout[::4] = True
    batch = batch._replace(context_weight=(~holdout).float(), loss_weight=holdout.float())

    def scale_params(b):
        torch.manual_seed(0)
        return model.scale_model(b, mc_samples=2, training=False)

    a = scale_params(batch)
    iobs = batch.iobs.clone()
    iobs[holdout] *= 100.
    b = scale_params(batch._replace(iobs=iobs, sigiobs=batch.sigiobs))
    assert torch.allclose(a[~holdout], b[~holdout])


def test_validation_metrics_come_from_held_out_rows():
    torch.manual_seed(0)
    model = build_small_model()
    batch = synthetic_batch(model).map(torch.as_tensor)
    with torch.no_grad(), tracking.track():
        model(batch, training=True)  # the standardization layers need moments
    model.eval()
    holdout = torch.zeros(len(batch.iobs), dtype=torch.bool)
    holdout[::5] = True
    with torch.no_grad(), tracking.track() as t:
        model(batch._replace(context_weight=(~holdout).float(), loss_weight=holdout.float()), training=False)
    assert torch.isfinite(t.total_loss())
    assert {'NLL', 'KL', 'CCpred'} <= set(t.metrics)


def test_loss_counts_only_weighted_rows():
    """With loss weight on a subset of rows, the objective is that subset's alone."""
    torch.manual_seed(0)
    model = build_small_model()
    batch = synthetic_batch(model).map(torch.as_tensor)
    with torch.no_grad(), tracking.track():
        model(batch, training=True)  # standardization moments
    model.eval()
    keep = torch.zeros(len(batch.iobs), dtype=torch.bool)
    keep[batch.image_id == 1] = True

    def nll(b):
        torch.manual_seed(0)
        with torch.no_grad(), tracking.track() as t:
            model(b, training=False)
        return float(t.metrics["NLL"])

    weighted = nll(batch._replace(loss_weight=keep.float()))
    # Image 1 alone, scored with the same context: drop the other images' loss only
    assert weighted != nll(batch)
    # Changing an unweighted row's intensity cannot change the weighted loss
    # unless it feeds the shared context -- give it no context either.
    iobs = batch.iobs.clone()
    iobs[~keep] *= 3.
    a = nll(batch._replace(loss_weight=keep.float(), context_weight=keep.float()))
    b = nll(batch._replace(iobs=iobs, loss_weight=keep.float(), context_weight=keep.float()))
    assert a == pytest.approx(b, rel=1e-5)
