import pytest
import numpy as np
import tensorflow as tf
from abismal.layers import ResNetDense



@pytest.mark.parametrize('dropout', [None, 0.1])
@pytest.mark.parametrize('hidden_units', [None, 12])
def test_resnet_ragged(dropout, hidden_units):
    data = np.random.random((10*100, 5)).astype('float32')
    rows = np.sort(np.random.choice(10, size=len(data)))
    ragged = tf.RaggedTensor.from_value_rowids(data, rows)
    rn = ResNetDense(hidden_units=hidden_units, dropout=dropout)
    out = rn(ragged)

@pytest.mark.parametrize('dropout', [None, 0.1])
@pytest.mark.parametrize('hidden_units', [None, 12])
def test_resnet(dropout, hidden_units):
    data = np.random.random((10, 100, 5)).astype('float32')
    rn = ResNetDense(hidden_units=hidden_units, dropout=dropout)
    out = rn(data)

