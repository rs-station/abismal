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



class SpreadGP(tfk.models.Model):
    def __init__(self, wavelength_range, num_atoms, num_points=100, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)
        epsilon = 1e-3
        small = 0.1
        self.num_atoms = num_atoms
        self.num_points = num_points
        self.epsilon = epsilon
        self.wav_min,self.wav_max = wavelength_range
        init = tfk.initializers.RandomUniform(0., 1.)
        self.inducing_x = self.add_weight('x', shape=(num_atoms, num_points), initializer=init)
        self.inducing_fp = self.add_weight('fp', shape=(num_atoms, num_points), initializer='zeros')
        self.inducing_fpp = self.add_weight('fpp', shape=(num_atoms, num_points), initializer='zeros')
        self.inducing_s = tfu.TransformedVariable(
            tf.eye(num_points, batch_shape=(num_atoms,)),
            tfb.Chain([
                tfb.CholeskyOuterProduct(),
                tfb.FillScaleTriL(
                    diag_bijector=tfb.Chain([
                        tfb.Shift(epsilon),
                        tfb.Exp(),
                    ])
                ),
            ]),
        )

        #self.jitter = tfu.TransformedVariable(
        #    small,
        #    tfb.Chain([
        #        tfb.Shift(epsilon),
        #        tfb.Exp(),
        #    ]),
        #)
        #self.jitter = epsilon
        self.jitter=0.
        self.bw = tfu.TransformedVariable(
            small,
            tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ]),
        )
        self.kfunc = tfm.psd_kernels.ExponentiatedQuadratic(
            length_scale = self.bw
        ) 

    def _get_variational_gps(self, wav):
        X = (wav - self.wav_min) / (self.wav_max - self.wav_min)
        qp = tfd.VariationalGaussianProcess(
              self.kfunc,
              X[None,None,...],
              self.inducing_x[None,...,None],
              self.inducing_fp[None,...],
              self.inducing_s[None,...],
              observation_noise_variance=0.,
              jitter=self.jitter,
        )
        qpp = tfd.VariationalGaussianProcess(
              self.kfunc,
              X[None,None,...],
              self.inducing_x[None,...,None],
              self.inducing_fpp[None,...],
              self.inducing_s[None,...],
              observation_noise_variance=0.,
              jitter=self.jitter,
        )
        return qp, qpp

    def call(self, wav):
        qp, qpp = self._get_variational_gps(wav)

        fp = tf.transpose(tf.squeeze(qp.mean(), axis=0))
        fpp = tf.transpose(tf.squeeze(qpp.mean(), axis=0))
        scale = tf.transpose(tf.squeeze(qp.stddev(), axis=0))
        self.add_metric(self.bw, "BW")
        self.add_metric(self.jitter, "Jitter")

#        kl_div = tf.reduce_mean(
#            qp.surrogate_posterior_kl_divergence_prior()
#        ) + tf.reduce_mean(
#            qpp.surrogate_posterior_kl_divergence_prior()
#        )
#
#        self.add_metric(kl_div, "KL")
#        self.add_loss(kl_div)

        return fp, fpp, scale

#    def compute_kl_terms(self, q, p, samples=None):
#        return None

