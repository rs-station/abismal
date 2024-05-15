import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.structure_factor.wilson import WilsonPrior


class FoldedNormalPosterior(StructureFactorPosteriorBase):
    def __init__(self, rac, scale_factor=1e-1, epsilon=1e-12, kl_weight=1., **kwargs):
        super().__init__(rac, epsilon=epsilon, kl_weight=kl_weight, **kwargs)
        self.low = self.epsilon * tf.cast(~self.rac.centric, dtype='float32')
        p = self.flat_prior()
        self.loc = tf.Variable(p.mean())
        self.scale = tfu.TransformedVariable(
            scale_factor * p.stddev(),
            tfb.Chain([
                tfb.Shift(epsilon), 
                tfb.Exp(),
            ]),
        )

    def flat_prior(self):
        prior = WilsonPrior(
            self.rac.centric,
            self.rac.epsilon,
        )
        return prior

    def flat_distribution(self):
        f = FoldedNormal(
            self.loc, 
            self.scale, 
        )
        q = tfd.TransformedDistribution(
            f, 
            tfb.Shift(self.low),
        )
        return q
