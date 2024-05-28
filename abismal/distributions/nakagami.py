import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


n = 100
conc,rate = np.random.random((2, n))
q = tfd.Gamma(conc, rate)


def moments_sqrt(q):
    alpha = q.concentration
    beta  = q.rate
    m = alpha
    omega = alpha / beta
    mean = tf.math.exp(
        tf.math.lgamma(m + 0.5) - tf.math.lgamma(m) + 0.5 * tf.math.log(omega) - 0.5 * tf.math.log(m)
    )
    var = omega - mean*mean
    std = tf.math.sqrt(var)
    return mean,std


s = 10_000
f,sigf = moments_sqrt(q)

z = q.sample(s)
z = tf.sqrt(z)
f_mc = tf.reduce_mean(z, axis=0)
sigf_mc = tf.math.reduce_std(z, axis=0)
from IPython import embed
embed(colors='linux')
