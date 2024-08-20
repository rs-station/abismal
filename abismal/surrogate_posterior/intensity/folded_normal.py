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
        self.low = self.epsilon 
        p = self.prior(self.rac.asu_id[...,None], self.rac.Hunique)
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

    def prior(self, asu_id, hkl):
        p = WilsonPrior(
            self.rac.gather(self.rac.centric, asu_id, hkl),
            self.rac.gather(self.rac.epsilon, asu_id, hkl),
        )
        return p

    def _distribution(self, loc, scale, low):
        f = FoldedNormal(
            loc, 
            scale, 
        )
        q = tfd.TransformedDistribution(
            f, 
            tfb.Shift(self.low),
        )
        return q

    def distribution(self, asu_id, hkl):
        loc = self.rac.gather(self.loc, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        q = self._distribution(loc, scale, self.low)
        return q

    def flat_distribution(self):
        q = self._distribution(self.loc, self.scale, self.low)
        return q

