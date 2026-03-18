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





class SpreadNN(tfk.models.Model):
    def __init__(self, wavelength_range, num_atoms, epsilon=1e-12, dmodel=32, mlp_depth=5, activation='swish', gated=True, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = 1e-5
        self.num_atoms = num_atoms
        self.epsilon = epsilon
        self.wav_min,self.wav_max = wavelength_range
        self.input_layer = tfk.layers.Dense(dmodel, kernel_initializer='glorot_normal')
        self.mlp = MLP(depth=mlp_depth, activation=activation, gated=gated)
        self.output_layer = tfk.layers.EinsumDense(
            '...d,dab->...ab',
            output_shape=(3, self.num_atoms),
            kernel_initializer='glorot_normal',
            bias_axes='ab',
        )

    def scale_bijector(self, x):
        return tf.nn.softplus(x) + self.epsilon

    def encode_wav(self, wav):
        out = 2. * (wav - self.wav_min) / (self.wav_max - self.wav_min) - 1.
        f = 2. * np.pi * 2 ** tf.linspace(0., 5., 6)
        out = tf.concat((
            tf.math.cos(out * f),
            tf.math.sin(out * f),
        ), axis=-1)
        return out

    def call(self, wav):
        wav_normed = self.encode_wav(wav)
        out = self.input_layer(wav_normed)
        out = self.mlp(out)
        out = self.output_layer(out)
        fp,fpp,scale = tf.unstack(out, axis=-2)
        scale = self.scale_bijector(scale)

        qp = tfd.Normal(fp, scale)
        qpp = tfd.Normal(fpp, scale)
        p = tfd.Normal(0., 1.)

        kl_div = tf.reduce_mean(
            qp.kl_divergence(p) + qpp.kl_divergence(p)
        )
        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(kl_div, 'Custom_KL')
        return fp, fpp, scale

    def compute_kl_terms(self, q, p, samples=None):
        return None

