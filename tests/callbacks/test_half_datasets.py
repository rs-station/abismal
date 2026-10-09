"""Half-dataset posteriors and the CC1/2 / CCanom computed from them."""
import numpy as np
import pandas as pd
import pytest
import torch

from abismal import tracking
from abismal.callbacks.half_datasets import (
    HalfDatasetStatistics, half_dataset_statistics, half_datasets, pair_datasets, weighted_pearson,
)
from abismal.training import Trainer, parameter_groups
from abismal.optimizers import Adam
from model_helpers import build_small_model, synthetic_batch


def _stats(model, bins=10):
    (ds_a, ds_b), = half_datasets(model)
    return half_dataset_statistics(*pair_datasets(ds_a, ds_b), bins=bins)


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
    overall, binned = _stats(model, bins=4)
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
    overall, _ = _stats(model)
    assert abs(overall["CC½"]) < 0.2
    assert abs(overall["CCanom"]) < 0.2


def test_only_reflections_seen_by_both_halves_count():
    model = build_small_model(split_halves=True)
    n = model.surrogate_posterior.rac.asu_size
    _set_halves(model, torch.rand(n) + 0.5, torch.rand(n) + 0.5)
    with torch.no_grad():
        model.half_posterior.seen[:n // 2] = False
    overall, _ = _stats(model)
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


def test_callback_writes_half_merges_in_cchalf_layout(tmp_path):
    model = build_small_model(split_halves=True)
    n = model.surrogate_posterior.rac.asu_size
    _set_halves(model, torch.rand(n) + 0.5, torch.rand(n) + 0.5)

    class _Trainer:
        history = {}
    callback = HalfDatasetStatistics(str(tmp_path), bins=3)
    callback.model, callback.trainer = model, _Trainer()
    logs = {}
    callback.on_epoch_end(0, logs)
    for half in (1, 2):
        assert (tmp_path / f"half_{half}" / "asu_0_epoch_1.mtz").exists()
    assert "CCanom" in logs and "CC½" in callback.trainer.history
    binned = pd.read_csv(tmp_path / "half_dataset_statistics.csv")
    assert list(binned.Epoch.unique()) == [1] and len(binned) == 3


def test_statistics_from_written_files_match_in_memory(tmp_path):
    """The files a run writes give the same numbers it logged."""
    import reciprocalspaceship as rs
    model = build_small_model(split_halves=True)
    n = model.surrogate_posterior.rac.asu_size
    g = torch.Generator().manual_seed(1)
    f = torch.rand(n, generator=g) + 0.5
    _set_halves(model, f, f + 0.1 * torch.rand(n, generator=g))
    (a, b), = half_datasets(model)
    a.write_mtz(str(tmp_path / "a.mtz")); b.write_mtz(str(tmp_path / "b.mtz"))
    mem, _ = half_dataset_statistics(*pair_datasets(a, b))
    disk, _ = half_dataset_statistics(*pair_datasets(rs.read_mtz(str(tmp_path / "a.mtz")), rs.read_mtz(str(tmp_path / "b.mtz"))))
    for k in mem:
        assert disk[k] == pytest.approx(mem[k], rel=1e-5)
