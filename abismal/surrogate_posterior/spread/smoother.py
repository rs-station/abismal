import numpy as np
import math
import gemmi
import tensorflow as tf
from abismal.distributions import Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.layers import MLP
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.symmetry import ReciprocalASU,ReciprocalASUCollection,ReciprocalASUGraph
import reciprocalspaceship as rs
from tempfile import NamedTemporaryFile
from subprocess import call

class SpreadSmoother(tfk.models.Model):
    def __init__(self, wavelength_range, num_atoms, num_points=100, epsilon=1e-12, train_inducing_x=False, train_bw=False, kl_weight=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.num_atoms = num_atoms
        self.kl_weight = kl_weight
        self.num_points = num_points
        self.epsilon = epsilon
        self.wav_min,self.wav_max = wavelength_range
        if train_inducing_x:
            init = tfk.initializers.RandomUniform(self.wav_min, self.wav_max)
            self.inducing_x = self.add_weight('x', shape=(num_atoms, num_points), initializer=init, trainable=train_inducing_x)
        else:
            self.inducing_x = tf.linspace(self.wav_min, self.wav_max, num_points)[None,:]
        self.inducing_fp = self.add_weight('fp', shape=(num_atoms, num_points), initializer='zeros')
        self.inducing_fpp = self.add_weight('fpp', shape=(num_atoms, num_points), initializer='zeros')

        self.inducing_s = tfu.TransformedVariable(
            1e-3 * tf.ones((num_atoms, num_points)),
            tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ]),
        )

        #bw = (self.wav_max - self.wav_min) / 20.
        bw = (self.wav_max - self.wav_min) / 10
        if train_bw:
            self.bw = tfu.TransformedVariable(
                bw,
                tfb.Chain([
                    tfb.Shift(epsilon),
                    tfb.Exp(),
                ]),
            )
        else:
            self.bw = bw

    def call(self, wav, dHKL=None):
        self.add_metric(self.bw, "BW")
        self.add_metric(tf.math.reduce_std(self.inducing_x), "SigX")
        d = tf.square((wav[:,None,...] - self.inducing_x[None,...]) / self.bw)
        w = tf.nn.softmax(-d, axis=-1)
        fp = tf.einsum('...d,...d->...', self.inducing_fp, w)
        fpp = tf.einsum('...d,...d->...', self.inducing_fpp, w)
        #scale  = tf.math.sqrt(tf.einsum('...d,...d->...', self.inducing_s, w * w))
        scale  = tf.einsum('...d,...d->...', self.inducing_s, w)

        p = tfd.Normal(0., 1.)
        kl = 0.5 * tf.reduce_mean(
            tfd.Normal(fp, scale).kl_divergence(p)
        ) + 0.5 * tf.reduce_mean(
            tfd.Normal(fpp, scale).kl_divergence(p)
        )
        self.add_metric(kl, name='CustomKL')
        self.add_loss(self.kl_weight * kl)

        return fp, fpp, scale


