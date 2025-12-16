import gemmi
import pytest
import tensorflow as tf
from abismal.likelihood import NormalLikelihood, StudentTLikelihood


@pytest.mark.parametrize('df', [None, 32.])
def test_likelihood(df):
    if df is None:
        likelihood = NormalLikelihood()
    else:
        likelihood = StudentTLikelihood(df)

    eps = 1e-3
    mc_samples = 5
    refls = 10
    ipred = tf.random.uniform((refls, mc_samples))
    iobs, sigiobs = tf.unstack(tf.random.uniform((2, refls, 1)), axis=0)
    l = likelihood(ipred, iobs, sigiobs)
    assert tf.math.reduce_all(tf.math.is_finite(l))

