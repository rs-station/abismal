import gemmi
import numpy as np
import pytest
import tensorflow as tf

from abismal.symmetry import ReciprocalASU, ReciprocalASUCollection, merge_dmin


SWAP = "-y,-x,-z"

# a and b differ by 3.5%, close enough that gemmi.find_twin_laws offers the h/k
# swap at its default obliquity but far enough that the swap moves reflections
# in resolution.
PSEUDO_TETRAGONAL = (74.008, 76.617, 145.826, 90.0, 90.0, 90.0)


def test_identity_needs_no_padding():
    cell = gemmi.UnitCell(*PSEUDO_TETRAGONAL)
    assert merge_dmin(cell, ["x,y,z"], 1.8) == 1.8


def test_true_lattice_symmetry_needs_no_padding():
    """a == b makes the swap an exact lattice symmetry, so nothing moves."""
    cell = gemmi.UnitCell(78.0, 78.0, 37.0, 90.0, 90.0, 90.0)
    assert merge_dmin(cell, ["x,y,z", SWAP], 1.8) == pytest.approx(1.8)


def test_pseudo_symmetry_pads_by_the_axial_ratio():
    """For an h/k swap the worst contraction is just a/b."""
    cell = gemmi.UnitCell(*PSEUDO_TETRAGONAL)
    expected = 1.8 * cell.a / cell.b
    assert merge_dmin(cell, ["x,y,z", SWAP], 1.8) == pytest.approx(expected)


def test_accepts_gemmi_ops():
    cell = gemmi.UnitCell(*PSEUDO_TETRAGONAL)
    triplets = merge_dmin(cell, ["x,y,z", SWAP], 1.8)
    ops = merge_dmin(cell, [gemmi.Op("x,y,z"), gemmi.Op(SWAP)], 1.8)
    assert triplets == pytest.approx(ops)


def test_padded_asu_contains_every_reindexed_reflection():
    """The invariant the padding exists for.

    Every reflection the loader will admit must have an entry in the ASU under
    every operator, or the gather silently returns reflection 0 in its place.
    """
    cell = gemmi.UnitCell(*PSEUDO_TETRAGONAL)
    sg = gemmi.SpaceGroup("P 21 21 21")
    dmin, ops = 1.8, ["x,y,z", SWAP]

    rac = ReciprocalASUCollection(
        ReciprocalASU(cell, sg, merge_dmin(cell, ops, dmin), anomalous=True)
    )

    # Every miller index the loader would keep, not just the ASU reps: inside
    # dmin, not systematically absent, not the origin.
    from reciprocalspaceship.utils import is_absent

    h, k, l = np.array(cell.get_hkl_limits(dmin))
    H = np.mgrid[-h:h + 1, -k:k + 1, -l:l + 1].reshape((3, -1)).T
    H = H[cell.calculate_d_array(H) >= dmin]
    H = H[np.any(H != 0, axis=1)]
    H = H[~is_absent(H, sg)]

    for triplet in ops:
        op = gemmi.Op(triplet)
        image = np.array([op.apply_to_hkl(tuple(x)) for x in H], dtype=np.int32)
        asu_id = np.zeros((len(image), 1), dtype=np.int32)
        misses = 0
        for a, x in zip(np.array_split(asu_id, 20), np.array_split(image, 20)):
            ids = rac._miller_ids(a, tf.convert_to_tensor(np.ascontiguousarray(x)))
            misses += int((ids.numpy() < 0).sum())
        assert misses == 0, f"{misses} reflections fall outside the ASU under {triplet}"


def test_unpadded_asu_does_lose_reflections():
    """Guards the test above: without padding the invariant really does fail."""
    cell = gemmi.UnitCell(*PSEUDO_TETRAGONAL)
    sg = gemmi.SpaceGroup("P 21 21 21")
    rac = ReciprocalASUCollection(ReciprocalASU(cell, sg, 1.8, anomalous=True))

    H = rac.reciprocal_asus[0].Hunique
    op = gemmi.Op(SWAP)
    image = np.array([op.apply_to_hkl(tuple(x)) for x in H], dtype=np.int32)
    asu_id = np.zeros((len(image), 1), dtype=np.int32)
    ids = np.concatenate([
        rac._miller_ids(a, tf.convert_to_tensor(np.ascontiguousarray(x))).numpy()
        for a, x in zip(np.array_split(asu_id, 20), np.array_split(image, 20))
    ])
    assert (ids < 0).any()


def test_valid_flags_out_of_asu_indices():
    cell = gemmi.UnitCell(*PSEUDO_TETRAGONAL)
    sg = gemmi.SpaceGroup("P 21 21 21")
    rac = ReciprocalASUCollection(ReciprocalASU(cell, sg, 1.8, anomalous=True))

    inside = np.array([[10, 10, 10], [5, 4, 20]], dtype=np.int32)
    # k well past h at the edge: the swap pushes these beyond dmin.
    outside = np.array([[0, 42, 0]], dtype=np.int32)
    H = np.concatenate([inside, outside])
    op = gemmi.Op(SWAP)
    image = np.array([op.apply_to_hkl(tuple(x)) for x in H], dtype=np.int32)

    splits = tf.constant([0, len(H)], dtype=tf.int64)
    hkl = tf.RaggedTensor.from_row_splits(tf.convert_to_tensor(image), splits)
    asu_id = tf.RaggedTensor.from_row_splits(tf.zeros((len(H), 1), "int32"), splits)

    valid = rac.valid(asu_id, hkl).flat_values.numpy().ravel()
    assert valid.tolist() == [True, True, False]

    # ... and the gather it protects would have returned reflection 0.
    ids = rac._miller_ids(tf.zeros((len(H), 1), "int32"), tf.convert_to_tensor(image))
    assert ids.numpy()[-1] == -1
