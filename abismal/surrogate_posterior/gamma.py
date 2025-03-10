import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior import StructureFactorPosteriorBase



class GammaPosteriorBase(object):
    """
    A base class for creating gamma posteriors. 
    """
    def __init__(self, rac, loc_init=None, scale_init=None,  epsilon=1e-12, concentration_min=0., **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.rac = rac

        #For serialization
        if loc_init is None:
            loc_init = tf.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * loc_init

        conc_init = tf.square(loc_init / scale_init)
        rate_init = conc_init / loc_init

        self.rate = tfu.TransformedVariable(
            rate_init,
            tfb.Chain([
                tfb.Shift(epsilon), 
                tfb.Exp(),
            ]),
        )

        #Concentration should remain above one to prevent change in curvature
        self.concentration = tfu.TransformedVariable(
            conc_init,
            tfb.Chain([
                tfb.Shift(concentration_min + epsilon), 
                tfb.Exp(),
            ]),
        )
        self.built=True

    def _distribution(self, concentration, rate):
        return tfd.Gamma(concentration, rate)

    def distribution(self, asu_id, hkl):
        rate = self.rac.gather(self.rate, asu_id, hkl)
        concentration = self.rac.gather(self.concentration, asu_id, hkl)
        q = self._distribution(concentration, rate)
        return q

    def flat_distribution(self):
        q = self._distribution(self.concentration, self.rate)
        return q

    def distribution(self, asu_id, hkl):
        concentration = self.rac.gather(self.concentration, asu_id, hkl)
        rate = self.rac.gather(self.rate, asu_id, hkl)
        q = self._distribution(concentration, rate)
        return q

