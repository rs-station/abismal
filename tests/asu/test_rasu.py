import pytest
import gemmi
from abismal.asu import ReciprocalASU,ReciprocalASUCollection
import tensorflow as tf
import numpy as np
from reciprocalspaceship.utils import is_absent,hkl_to_asu

def get_rasu_test_data(sg, cell, dmin, anomalous):
    rasu = ReciprocalASU(cell, sg, dmin, anomalous)
    go = sg.operations()

    h,k,l = rasu.hmax
    H = np.mgrid[-h:h+1:1,-k:k+1:1,-l:l+1:1].reshape((3, -1)).T

    #Apply resolution cutoff and remove absences
    d = cell.calculate_d_array(H)
    H = H[d >= dmin]
    H = H[~is_absent(H, sg)]

    #Remove 0, 0, 0
    H = H[np.any(H != 0, axis=1)]

    Hasu,Isym = hkl_to_asu(H, sg)
    if anomalous:
        friedel_sign = np.array([-1, 1])[Isym % 2][:,None]
        centric = go.centric_flag_array(Hasu).astype(bool)
        friedel_sign[centric] = 1
        Hasu = friedel_sign*Hasu

    Hunique,inv = np.unique(Hasu, axis=0, return_inverse=True)
    miller_ids = np.arange(inv.max() + 1)[inv].astype(np.int32)

    return rasu, H, miller_ids

@pytest.mark.parametrize('anomalous', [False, True])
def test_reciprocal_asu(anomalous):
    sg = gemmi.SpaceGroup(19)
    cell = gemmi.UnitCell(10., 20., 30., 90., 90., 90.)
    dmin = 2.

    rasu, H, miller_ids = get_rasu_test_data(sg, cell, dmin, anomalous)

    assert rasu._miller_ids(H[0]) == miller_ids[0]

    assert np.array_equal(rasu._miller_ids(H).numpy() , miller_ids)

    test = rasu.gather(
        np.arange(rasu.asu_size).astype('int32'),
        H
    )
    assert np.array_equal(test, miller_ids)

    #Test out of bounds indices
    assert rasu._miller_ids(rasu.hmax + 1) == -1

    #Test ragged miller ids
    n = len(H)
    image_id = np.sort(np.random.choice(10, n)).astype(np.int32)
    ragged_H = tf.RaggedTensor.from_value_rowids(H, image_id)
    test = rasu._miller_ids(ragged_H)
    expected = tf.RaggedTensor.from_value_rowids(miller_ids, image_id)
    assert tf.reduce_all(expected == test)

    #Test ragged gather
    test = rasu.gather(np.arange(rasu.asu_size, dtype='int32'), ragged_H)
    assert tf.reduce_all(expected == test)

@pytest.mark.parametrize('anomalous', [False, True])
def test_reciprocal_asu_collection(anomalous):
    sg = gemmi.SpaceGroup(19)
    cell = gemmi.UnitCell(10., 20., 30., 90., 90., 90.)

    dmin = 1.8
    rasu_1,H_1,miller_id_1 = get_rasu_test_data(sg, cell, dmin, anomalous)

    sg = gemmi.SpaceGroup(4)
    dmin = 2.4
    rasu_2,H_2,miller_id_2 = get_rasu_test_data(sg, cell, dmin, anomalous)

    rac = ReciprocalASUCollection(rasu_1, rasu_2)

    
    n = 100
    asu_ids = np.random.choice(2, size=n)
    H = np.zeros((n, 3), dtype='int32')
    miller_ids = np.zeros(n, dtype='int32')

    n1 = np.sum(asu_ids == 0)
    n2 = np.sum(asu_ids == 1)

    idx1 = np.random.choice(len(H_1), size=n1)
    idx2 = np.random.choice(len(H_2), size=n2)

    H[asu_ids == 0] = H_1[idx1]
    H[asu_ids == 1] = H_2[idx2]

    miller_ids[asu_ids == 0] = miller_id_1[idx1]
    miller_ids[asu_ids == 1] = miller_id_2[idx2]

    #Test floats
    test = rac.gather(
        (np.arange(rasu_1.asu_size, dtype='float32'), np.arange(rasu_2.asu_size, dtype='float32')),
        asu_ids, H
    )
    assert np.array_equal(test, miller_ids)

    #Test integers
    test = rac.gather(
        (np.arange(rasu_1.asu_size, dtype='int32'), np.arange(rasu_2.asu_size, dtype='int32')),
        asu_ids, H
    )
    assert np.array_equal(test, miller_ids)

    #Test ragged tensors
    image_id = np.sort(np.random.choice(3, size=n))
    ragged_H = tf.RaggedTensor.from_value_rowids(H, image_id)
    ragged_asu_ids = tf.RaggedTensor.from_value_rowids(asu_ids, image_id)
    test = rac.gather(
        (np.arange(rasu_1.asu_size, dtype='int32'), np.arange(rasu_2.asu_size, dtype='int32')),
        ragged_asu_ids, ragged_H
    )
    expected = tf.RaggedTensor.from_value_rowids(miller_ids, image_id)
    assert tf.reduce_all(expected == test)

