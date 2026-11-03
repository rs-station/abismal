"""
The per-image bookkeeping in VariationalMergingModel, which the TF version did
with ragged reductions and now does with index_add over image ids.
"""
import numpy as np
import pytest
import torch

from abismal import tracking
from abismal.merging import VariationalMergingModel
from model_helpers import build_small_model, synthetic_batch


def _ragged_reference(ll, valid, image_id, num_images):
    """What the TF code computed: per image, sum(ll * valid) / (nvalid * mc), or -1e30."""
    out = np.empty(num_images)
    for b in range(num_images):
        rows = image_id == b
        nvalid = valid[rows].sum()
        if nvalid == 0:
            out[b] = -1e30
        else:
            out[b] = (ll[rows] * valid[rows]).sum() / (nvalid * ll.shape[-1])
    return out


def test_image_log_likelihood_matches_the_ragged_reduction():
    rng = np.random.default_rng(0)
    image_id = np.repeat(np.arange(4), [3, 5, 2, 4])
    ll = rng.normal(size=(len(image_id), 6)).astype(np.float32)
    valid = rng.random((len(image_id), 1)) > 0.3
    valid[image_id == 2] = False  # an image nothing indexes

    model = build_small_model()
    got = model._image_log_likelihood(
        torch.as_tensor(ll), torch.as_tensor(valid), torch.as_tensor(image_id), 4
    ).numpy()
    expected = _ragged_reference(ll, valid, image_id, 4)
    assert np.allclose(got, expected, rtol=1e-5)
    assert got[2] == -1e30


def test_select_takes_the_winning_operator_per_image():
    model = build_small_model()
    image_id = torch.tensor([0, 0, 1, 1, 1, 2])
    best = torch.zeros(6, 3)
    candidate = torch.ones(6, 3)
    idx = torch.tensor([False, True, False])
    out = model._select(best, idx, image_id, candidate)
    assert torch.equal(out[:, 0], torch.tensor([0., 0., 1., 1., 1., 0.]))
    # 1d candidates (e.g. hkl rows) broadcast the same way
    hkl = model._select(torch.zeros(6, 3, dtype=torch.long), idx, image_id, torch.ones(6, 3, dtype=torch.long))
    assert torch.equal(hkl[:, 0], torch.tensor([0, 0, 1, 1, 1, 0]))


def test_reindexing_picks_the_consistent_operator():
    """
    With a posterior trained to one indexing, an image presented under the
    other indexing of a twin law must be scored under the operator that undoes
    it, and the reflections it marks seen must be the undone ones.
    """
    torch.manual_seed(0)
    # k,h,-l is its own inverse, so it undoes itself. It need not be a true
    # twin law of the cell for this test: it only has to be a candidate.
    model = build_small_model(reindexing_ops=["x,y,z", "k,h,-l"])
    rac = model.surrogate_posterior.rac
    # Distinctive structure factors so the two indexings disagree
    with torch.no_grad():
        model.surrogate_posterior.loc.raw.copy_(torch.randn(rac.asu_size) * 2.)
        model.surrogate_posterior.scale.raw.fill_(np.log(1e-3))
    F = model.surrogate_posterior.loc().detach()

    batch = synthetic_batch(model, num_images=2, refls_per_image=40, seed=1)
    batch = batch.map(torch.as_tensor)
    # Intensities that match the posterior exactly under the identity
    ids = rac._miller_ids(batch.asu_id, batch.hkl)
    iobs = (F[ids.clamp_min(0)] ** 2)[:, None]
    batch = batch._replace(iobs=iobs, sigiobs=0.01 * iobs + 0.01)
    # Present image 1 in the alternate indexing
    from abismal.symmetry import Op
    twin = Op("k,h,-l")
    flipped = torch.where((batch.image_id == 1)[:, None], twin(batch.hkl), batch.hkl)
    batch = batch._replace(hkl=flipped)

    model.standardize = False
    model.standardize_intensity = model.standardize_metadata = None
    model.scale_model.kl_weight = 0.
    with tracking.track():
        model(batch, training=True)
    seen = model.surrogate_posterior.seen
    # Everything seen came from the identity-indexed reflections
    assert seen[ids[ids >= 0]].all()
    assert int(seen.sum()) == len(torch.unique(ids[ids >= 0]))


def test_the_loss_weights_every_observation_once_whatever_image_it_is_on():
    # Images of 1 and 9 reflections: averaging each image first would give the lone reflection nine times the weight
    # of each of the others.
    model = build_small_model()
    ll = torch.tensor([[9.]] + [[0.]] * 9).repeat(1, 4)
    valid = torch.ones(10, 1, dtype=torch.bool)
    valid[-1] = False  # and an unindexable observation does not count
    assert torch.isclose(model._observation_mean(ll, valid), torch.tensor(1.))
    image_id = torch.tensor([0] + [1] * 9)
    assert torch.allclose(model._image_log_likelihood(ll, valid, image_id, 2), torch.tensor([9., 0.]))


def test_the_kl_of_an_independent_posterior_counts_valid_observations_only():
    model = build_small_model()
    kl = torch.tensor([[2.], [4.], [0.]])  # the third, unindexable, already masked to 0
    valid = torch.tensor([[True], [True], [False]])
    assert torch.isclose(model._observation_mean(kl, valid), torch.tensor(3.))


def _three_reflections(model):
    hkl = model.surrogate_posterior.rac.Hunique[:3].reshape(-1, 3)
    return torch.zeros(len(hkl), 1, dtype=torch.long), hkl


@pytest.mark.parametrize('mode, expected', [('standardized', [12. / 7., 6. / 7., 3. / 7.]), ('exact', [1., .5, .25])])
def test_kl_reflection_weights_are_inverse_counts(mode, expected):
    model = build_small_model(kl_reflection_weights=mode)
    asu, hkl = _three_reflections(model)
    assert torch.equal(model._kl_reflection_weight(asu, hkl).reshape(-1), torch.ones(3))  # nothing counted yet
    for rows in ([0, 1, 2], [1, 2], [2], [2]):  # counts of 1, 2 and 4
        model._count_observations(asu[rows], hkl[rows], torch.ones(len(rows), dtype=torch.bool))
    w = model._kl_reflection_weight(asu, hkl).reshape(-1)
    assert torch.allclose(w, torch.tensor(expected, dtype=w.dtype))


def test_observation_counts_skip_images_already_counted():
    model = build_small_model(kl_reflection_weights='exact')
    asu, hkl = _three_reflections(model)
    model._count_observations(asu, hkl, torch.tensor([True, False, True]))
    assert torch.equal(model.kl_counts[:3], torch.tensor([1., 0., 1.], dtype=model.kl_counts.dtype))


def test_kl_reflection_weights_reach_the_kl_of_a_forward_pass():
    from model_helpers import synthetic_batch
    model = build_small_model(kl_reflection_weights='exact')
    batch = synthetic_batch(model).map(torch.as_tensor)
    with tracking.track() as tracked:
        model(batch, training=True)
    assert model.kl_counts.sum() > 0  # training counted the batch
    assert tracked.metrics['KL'] > 0
    model._kl_reflection_weight = lambda asu_id, hkl: torch.zeros(len(hkl))
    with tracking.track() as tracked:
        model(batch, training=True)
    assert tracked.metrics['KL'] == 0
