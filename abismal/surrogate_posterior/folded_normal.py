import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior import StructureFactorPosteriorBase



class FoldedNormalPosteriorBase(object):
    """
    A base class for creating folded normal posteriors. 
    """
    def __init__(self, rac, loc_init, scale_init, epsilon=1e-12, **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.low = self.epsilon
        self._loc_init = loc_init
        self._scale_init = scale_init

        self.loc = tfu.TransformedVariable(
            loc_init,
            tfb.Exp(),
        )

        self.scale = tfu.TransformedVariable(
            scale_init,
            tfb.Chain([
                tfb.Shift(epsilon), 
                tfb.Exp(),
            ]),
        )
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({
            'loc_init' : self._loc_init,
            'scale_init' : self._scale_init,
        })
        return config

    def _distribution(self, loc, scale, low):
        f = FoldedNormal(
            loc, 
            scale, 
        )
        q = tfd.TransformedDistribution(
            f, 
            tfb.Shift(low),
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

