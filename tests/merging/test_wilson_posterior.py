import pytest
import numpy as np
import gemmi
import tensorflow as tf
from abismal.merging.surrogate_posterior import WilsonPosterior,PosteriorCollection
from abismal.asu import ReciprocalASU


@pytest.mark.parametrize("anomalous", [True, False])
def test_wilson_posterior_collection(anomalous):
    sg = gemmi.SpaceGroup(19)
    cell = gemmi.UnitCell(10., 20., 30., 90., 90., 90.)

    dmin = 1.8
    rasu_1 = ReciprocalASU(cell, sg, dmin, anomalous)

    sg = gemmi.SpaceGroup(4)
    dmin = 2.4
    rasu_2 = ReciprocalASU(cell, sg, dmin, anomalous)

    wp_1 = WilsonPosterior(rasu_1, 1.)
    wp_2 = WilsonPosterior(rasu_2, 1.)

    n = 100
    asu_id = np.random.choice(2, size=n)
    image_id = np.sort(np.random.choice(3, size=n))

    H1 = rasu_1.get_reciprocal_cell()
    H2 = rasu_2.get_reciprocal_cell()
    hkl = np.where(
        (asu_id == 0)[:,None],
        H1[np.random.choice(len(H1), size=n)],
        H2[np.random.choice(len(H2), size=n)],
    )

    hkl = tf.RaggedTensor.from_value_rowids(hkl, image_id)
    asu_id = tf.RaggedTensor.from_value_rowids(asu_id, image_id)

    wpc = PosteriorCollection(wp_1, wp_2)

    test = wpc(asu_id, hkl)

