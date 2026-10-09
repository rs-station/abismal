"""Half-dataset posteriors and the CC1/2 / CCanom computed from them."""
import numpy as np
import pytest
import torch

from abismal import tracking
from abismal.callbacks.half_datasets import half_dataset_statistics, weighted_pearson
from abismal.training import Trainer, parameter_groups
from abismal.optimizers import Adam
from model_helpers import build_small_model, synthetic_batch


def _set_halves(model, a, b):
    """Give the two half posteriors locations a and b, mark everything seen."""
    hp = model.half_posterior
    with torch.no_grad():
        hp.loc.raw.copy_(torch.log(torch.cat([a, b])))
        hp.scale.raw.fill_(np.log(1e-3))
        hp.seen.fill_(True)


def test_identical_halves_correlate_perfectly():
    model = build_small_model(split_halves=True)
    n = model.surrogate_posterior.rac.asu_size
    f = torch.rand(n) + 0.5
    _set_halves(model, f, f)
    overall, binned = half_dataset_statistics(model, bins=4)
    assert overall["CC½"] == pytest.approx(1.0)
    assert overall["CC½_spearman"] == pytest.approx(1.0)
    assert overall["CCanom"] == pytest.approx(1.0)
    assert overall["Nanom"] > 0
    assert len(binned) == 4
    assert np.allclose(binned["CC½"], 1.0)


def test_independent_halves_do_not_correlate():
    model = build_small_model(split_halves=True)
    n = model.surrogate_posterior.rac.asu_size
    g = torch.Generator().manual_seed(0)
    _set_halves(model, torch.rand(n, generator=g) + 0.5, torch.rand(n, generator=g) + 0.5)
    overall, _ = half_dataset_statistics(model)
    assert abs(overall["CC½"]) < 0.2
    assert abs(overall["CCanom"]) < 0.2


def test_only_reflections_seen_by_both_halves_count():
    model = build_small_model(split_halves=True)
    n = model.surrogate_posterior.rac.asu_size
    _set_halves(model, torch.rand(n) + 0.5, torch.rand(n) + 0.5)
    with torch.no_grad():
        model.half_posterior.seen[:n // 2] = False
    overall, _ = half_dataset_statistics(model)
    assert overall["N½"] == n - n // 2


def test_weighted_pearson_ignores_zero_weight_outlier():
    x = np.array([1., 2., 3., 4., 100.])
    y = np.array([1., 2., 3., 4., -100.])
    assert weighted_pearson(x, y, np.array([1., 1., 1., 1., 0.])) == pytest.approx(1.0)


def test_half_term_moves_only_the_half_posterior():
    """The halves are a probe: the full model must not feel them."""
    torch.manual_seed(0)
    model = build_small_model(split_halves=True)
    batch = synthetic_batch(model, num_images=4, split_halves=True).map(torch.as_tensor)
    model.train()
    with tracking.track() as tracker:
        model(batch, training=True)
    half_loss = tracker.losses[-1]  # the half-dataset term is added last
    assert "NLL½" in tracker.metrics
    grads = torch.autograd.grad(half_loss, list(model.parameters()), allow_unused=True)
    half_ids = {id(p) for p in model.half_posterior.parameters()}
    for p, g in zip(model.parameters(), grads):
        if id(p) in half_ids:
            continue
        assert g is None or torch.count_nonzero(g) == 0
    assert any(g is not None and torch.count_nonzero(g) > 0
               for p, g in zip(model.parameters(), grads) if id(p) in half_ids)


def test_each_half_sees_only_its_own_images():
    torch.manual_seed(0)
    model = build_small_model(split_halves=True)
    batch = synthetic_batch(model, num_images=2, split_halves=True)  # image 0 -> A, 1 -> B
    trainer = Trainer(model, Adam(parameter_groups(model)), verbose=0)
    trainer.train_step(batch.map(torch.as_tensor))
    n = model.surrogate_posterior.rac.asu_size
    rac = model.surrogate_posterior.rac
    ids = rac._miller_ids(torch.as_tensor(batch.asu_id), torch.as_tensor(batch.hkl)).numpy()
    seen = model.half_posterior.seen.numpy()
    a_ids = set(ids[batch.image_id == 0]); b_ids = set(ids[batch.image_id == 1])
    assert set(np.flatnonzero(seen[:n])) == a_ids
    assert set(np.flatnonzero(seen[n:])) == b_ids


def test_saving_round_trip_keeps_the_half_posterior(tmp_path):
    from abismal.saving import save_model, load_model
    model = build_small_model(split_halves=True)
    save_model(model, tmp_path / "m.pt")
    clone = load_model(tmp_path / "m.pt")
    assert clone.half_posterior is not None
    assert torch.equal(clone.half_posterior.loc.raw, model.half_posterior.loc.raw)
