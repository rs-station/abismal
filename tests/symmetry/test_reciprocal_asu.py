import gemmi
import numpy as np
import pytest
import torch

from abismal.symmetry import Op, ReciprocalASU, ReciprocalASUCollection, ReciprocalASUGraph


CELL = gemmi.UnitCell(34., 45., 98., 90., 90., 90.)
SG = gemmi.SpaceGroup("P 21 21 21")


@pytest.fixture
def rac():
    return ReciprocalASUCollection(
        ReciprocalASU(CELL, SG, 3.0, anomalous=True),
        ReciprocalASU(CELL, SG, 2.5, anomalous=False),
    )


def test_unique_reflections_map_to_themselves(rac):
    ids = rac._miller_ids(rac.asu_id[:, None], rac.Hunique)
    assert torch.equal(ids, torch.arange(rac.asu_size))


def test_symmetry_mates_share_an_id(rac):
    """Every symmetry equivalent of an ASU reflection looks up the same parameter."""
    go = SG.operations()
    H = rac.Hunique.numpy()
    a = rac.asu_id.numpy()[:, None]
    expected = np.arange(rac.asu_size)
    for op in go.sym_ops:
        image = np.array([op.apply_to_hkl(tuple(h)) for h in H])
        ids = rac._miller_ids(torch.as_tensor(a), torch.as_tensor(image)).numpy()
        # In the anomalous ASU an operator can produce the Friedel mate of an
        # acentric reflection only via inversion, which space-group ops lack.
        assert (ids == expected).all()


def test_friedel_mates(rac):
    H = rac.Hunique.numpy()
    a = rac.asu_id.numpy()[:, None]
    ids = rac._miller_ids(torch.as_tensor(a), torch.as_tensor(-H)).numpy()
    expected = np.arange(rac.asu_size)
    centric = rac.centric.numpy()
    anomalous = a[:, 0] == 0
    # The non-anomalous ASU merges Friedel mates, as do centrics.
    merged = ~anomalous | centric
    assert (ids[merged] == expected[merged]).all()
    assert (ids[~merged] != expected[~merged]).all()
    assert (ids >= 0).all()


def test_out_of_range_and_absent_are_invalid(rac):
    H = torch.tensor([[0, 0, 0], [1000, 0, 0], [0, 0, 1], [1, 2, 3]])
    a = torch.zeros((4, 1), dtype=torch.int64)
    # 0 0 0 is never stored, 1000 is off the grid, and 0 0 l is absent for odd l
    assert rac.valid(a, H)[:, 0].tolist() == [False, False, False, True]


def test_gather_matches_numpy_rasu(rac):
    rasu = rac.reciprocal_asus[0]
    rng = np.random.default_rng(0)
    H = rasu.Hall[rng.choice(len(rasu.Hall), 100)]
    values = rng.normal(size=rasu.asu_size).astype(np.float32)
    expected = rasu.gather(values, H)
    padded = np.concatenate([values, np.zeros(rac.asu_size - rasu.asu_size, np.float32)])
    a = torch.zeros((100, 1), dtype=torch.int64)
    out = rac.gather(torch.as_tensor(padded), a, torch.as_tensor(H)).numpy()
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("triplet", ["x,y,z", "-y,-x,-z", "k,h,-l", "-h,k,-l"])
def test_op_matches_gemmi(triplet):
    op = Op(triplet)
    H = np.random.default_rng(1).integers(-30, 30, size=(50, 3))
    expected = np.array([op.gemmi_op.apply_to_hkl(tuple(h)) for h in H])
    assert np.array_equal(op(H), expected)
    assert np.array_equal(op(torch.as_tensor(H)).numpy(), expected)
    assert np.array_equal(op(torch.as_tensor(H, dtype=torch.int32)).numpy(), expected)
