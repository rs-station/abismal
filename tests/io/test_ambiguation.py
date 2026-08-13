import pytest
import numpy as np
import tensorflow as tf
from abismal.io.ambiguation import Ambiguator


# P31 in a hexagonal cell has merohedral twin laws
AMBIGUOUS = ((79., 79., 38., 90., 90., 120.), "P31")


def hkl_dataset(length=64, hkl=(1., 2., 3.)):
    """ A dataset of `length` identical images, each with a single reflection """
    h = tf.constant([hkl], dtype='float32')
    z = tf.zeros([1, 1])
    def to_datum(i):
        return ((tf.zeros([1, 1], dtype='int32'), h, z, z, z, z, z), z)
    return tf.data.Dataset.range(length).map(to_datum)


def ambiguated_hkls(amb, length=64):
    data = hkl_dataset(length).enumerate().map(amb)
    return np.stack([X[1].numpy()[0] for X,Y in data])


def test_ambiguate_varies_across_images():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    assert len(amb.ops) > 1

    hkls = ambiguated_hkls(amb)
    assert len(np.unique(hkls, axis=0)) > 1


def test_ambiguate_fixed_across_passes():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    data = hkl_dataset().enumerate().map(amb)

    first  = np.stack([X[1].numpy()[0] for X,Y in data])
    second = np.stack([X[1].numpy()[0] for X,Y in data])
    assert np.array_equal(first, second)


def test_ambiguate_reproducible():
    """ A freshly built pipeline must reproduce the same assignment """
    first  = ambiguated_hkls(Ambiguator.from_symmetry(*AMBIGUOUS))
    second = ambiguated_hkls(Ambiguator.from_symmetry(*AMBIGUOUS))
    assert np.array_equal(first, second)


def test_ambiguate_seed_changes_assignment():
    amb = Ambiguator.from_symmetry(*AMBIGUOUS)
    other = Ambiguator(
        [op.gemmi_op.triplet() for op in amb.ops], seed=amb.seed + 1
    )
    assert not np.array_equal(ambiguated_hkls(amb), ambiguated_hkls(other))


def test_ambiguate_without_twin_laws():
    """ With only the identity op available, hkl passes through untouched """
    amb = Ambiguator(["x,y,z"])
    hkls = ambiguated_hkls(amb)
    assert np.array_equal(hkls, np.full_like(hkls, [1., 2., 3.]))
