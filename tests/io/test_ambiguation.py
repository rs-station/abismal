import pytest
import numpy as np
import gemmi
from abismal.io.ambiguation import Ambiguator, reindex


# P31 in a hexagonal cell has merohedral twin laws
AMBIGUOUS = ((79., 79., 38., 90., 90., 120.), "P31")


def hkl_dataset(length=64, hkl=(1, 2, 3)):
    """ `length` identical image records, each with a single reflection """
    h = np.array([hkl], dtype='int32')
    z = np.zeros((1, 1), dtype='float32')
    for _ in range(length):
        yield (np.zeros((1, 1), dtype='int32'), h.copy(), z, z, z, z, z)


def ambiguated_hkls(amb, length=64):
    data = (amb(i, x) for i, x in enumerate(hkl_dataset(length)))
    return np.stack([x[1][0] for x in data])


def test_ambiguate_varies_across_images():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    assert len(amb.ops) > 1

    hkls = ambiguated_hkls(amb)
    assert len(np.unique(hkls, axis=0)) > 1


def test_ambiguate_uses_every_op():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    chosen = [amb.choose(i) for i in range(1000)]
    counts = np.bincount(chosen, minlength=len(amb.ops))
    assert (counts > 0).all()
    assert len(counts) == len(amb.ops)


def test_ambiguate_fixed_across_passes():
    """ The op depends on the image's index alone, not on when it is asked """
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    first = ambiguated_hkls(amb)
    second = ambiguated_hkls(amb)
    assert np.array_equal(first, second)
    reversed_order = [amb.choose(i) for i in reversed(range(64))][::-1]
    assert reversed_order == [amb.choose(i) for i in range(64)]


def test_ambiguate_reproducible():
    """ A freshly built ambiguator must reproduce the same assignment """
    first  = ambiguated_hkls(Ambiguator.from_symmetry(*AMBIGUOUS))
    second = ambiguated_hkls(Ambiguator.from_symmetry(*AMBIGUOUS))
    assert np.array_equal(first, second)


def test_ambiguate_seed_changes_assignment():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    other = Ambiguator(
        [op.triplet() for op in amb.ops], seed=amb.seed + 1
    )
    assert not np.array_equal(ambiguated_hkls(amb), ambiguated_hkls(other))


def test_ambiguate_without_twin_laws():
    """ With only the identity op available, hkl passes through untouched """
    amb = Ambiguator(["x,y,z"])
    hkls = ambiguated_hkls(amb)
    assert np.array_equal(hkls, np.full_like(hkls, [1, 2, 3]))


def test_ambiguate_leaves_other_fields():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    record = next(hkl_dataset(1))
    out = amb(5, record)
    assert len(out) == 7
    assert out[1].dtype == np.int32
    for i in (0, 2, 3, 4, 5, 6):
        assert out[i] is record[i]


@pytest.mark.parametrize("triplet", ["x,y,z", "-x,-y,-z", "y,x,-z", "-x,-x+y,-z", "x-y,y,-z"])
def test_reindex_matches_gemmi(triplet):
    """ Reindexing a miller index is gemmi's apply_to_hkl """
    op = gemmi.Op(triplet)
    rng = np.random.default_rng(0)
    hkl = rng.integers(-20, 20, (100, 3)).astype('int32')
    out = reindex(hkl, op)
    assert out.dtype == np.int32
    expected = np.array([op.apply_to_hkl(list(map(int, h))) for h in hkl])
    assert np.array_equal(out, expected)
