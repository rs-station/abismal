import pytest
import numpy as np
import tensorflow as tf
from abismal.layers import ConvexCombination,ConvexCombinations



@pytest.mark.parametrize('dropout', [None, 0.1])
def test_convex_combination_ragged(dropout):
    data = np.random.random((10*100, 5)).astype('float32')
    rows = np.sort(np.random.choice(10, size=len(data)))
    ragged = tf.RaggedTensor.from_value_rowids(data, rows)
    cc = ConvexCombination(dropout=dropout)
    out = cc(ragged)
    out + data 
    expected_shape = list(ragged.shape)
    expected_shape[-2] = 1

    assert out.shape[-2] == 1
    assert np.array_equal(expected_shape, out.shape)


@pytest.mark.parametrize('dropout', [None, 0.1])
def test_convex_combination(dropout):
    data = np.random.random((10, 100, 5)).astype('float32')
    cc = ConvexCombination(dropout=dropout)
    out = cc(data)
    out + data 
    expected_shape = list(data.shape)
    expected_shape[-2] = 1

    assert out.shape[-2] == 1
    assert np.array_equal(expected_shape, out.shape)



@pytest.mark.parametrize('dropout', [None, 0.1])
@pytest.mark.parametrize('npoints', [1, 5])
def future_test_convex_combinations_ragged(dropout, npoints):
    """
    Currently ragged tensors are not supported for this layer
    """
    data = np.random.random((10*100, 5)).astype('float32')
    rows = np.sort(np.random.choice(10, size=len(data)))
    ragged = tf.RaggedTensor.from_value_rowids(data, rows)
    cc = ConvexCombinations(npoints, dropout=dropout)
    out = cc(ragged)
    expected_shape = list(ragged.shape)
    expected_shape[-2] = npoints
    assert np.array_equal(expected_shape, out.shape)


@pytest.mark.parametrize('dropout', [None, 0.1])
@pytest.mark.parametrize('npoints', [1, 5])
def test_convex_combinations(dropout, npoints):
    data = np.random.random((10, 100, 5)).astype('float32')
    cc = ConvexCombinations(npoints, dropout=dropout)
    out = cc(data)
    expected_shape = list(data.shape)
    expected_shape[-2] = npoints
    assert np.array_equal(expected_shape, out.shape)
