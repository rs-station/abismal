import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm
import numpy as np



vmin,vmax = -10, 10
n = vmax - vmin + 1
scale_init = loc_init = np.logspace(vmin, vmax, n, base=2)
loc = (loc_init[:,None] * np.ones((n, n))).flatten()
scale = (scale_init[None,:] * np.ones((n, n))).flatten()

conc = tf.square(loc_init[...,None] / scale_init[...,None,:])
rate = loc_init[...,None] / tf.square(scale_init[...,None,:])

rate = tf.reshape(rate, n * n)
conc = tf.reshape(conc, n * n)


q = tfd.Gamma(conc, rate)

assert np.allclose(q.mean(), loc)
assert np.allclose(q.stddev(), scale)


def moments_sqrt(q):
    conc = q.concentration
    rate  = q.rate
    omega = conc / rate

    ldiff = -tfm.log_gamma_difference(0.5, conc)
    mean = tf.math.exp(
        ldiff - 0.5*tf.math.log(rate) 
    )
    #var = omega - mean*mean
    var = (conc - tf.math.exp(2.* ldiff)) / rate
    std = tf.math.sqrt(var)
    return mean,std


s = 10_000
f,sigf = moments_sqrt(q)

z = q.sample(s)
z = tf.sqrt(z)
f_mc = tf.reduce_mean(z, axis=0)
sigf_mc = tf.math.reduce_std(z, axis=0)

eps = 1e-3
assert np.corrcoef(f, f_mc)[0,1] > 1. - eps
assert np.corrcoef(sigf, sig_mc)[0,1] > 1. - eps

from IPython import embed
embed(colors='linux')
