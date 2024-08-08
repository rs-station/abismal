import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.intensity.wilson import WilsonPrior


@tfk.saving.register_keras_serializable(package="abismal")
class FoldedNormalPosterior(IntensityPosteriorBase):
    def __init__(self, rac, scale_factor=1e-1, epsilon=1e-12, kl_weight=1., **kwargs):
        super().__init__(rac, epsilon=epsilon, kl_weight=kl_weight, **kwargs)
        self.low = self.epsilon * tf.cast(self.rac.centric, dtype='float32')
        p = self.flat_prior()
        loc_init = p.mean()
        self.loc = tf.Variable(loc_init)
        self.scale = tfu.TransformedVariable(
            scale_factor * loc_init,
            tfb.Chain([
                tfb.Shift(epsilon), 
                tfb.Exp(),
            ]),
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : self.rac,
            'scale_factor' : self._init_scale_factor,
            'epsilon' : self.epsilon,
            'kl_weight' : self.kl_weight,
        })
        return config

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

