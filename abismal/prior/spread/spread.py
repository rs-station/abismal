import numpy as np
import tensorflow as tf
import math
from abismal.distributions import FoldedNormal,Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection
from abismal.prior.base import PriorBase



class SpreadPrior(PriorBase):
    def __init__(self, rac, Fcalc, sigma=1.):
        super().__init__()
        self.rac = rac
        #self.sites = sites
        self.Fc = Fcalc
        self.sigma = sigma

    @classmethod
    def from_spread_posterior(cls, spread_posterior, sigma=1.):
        """
        Assumptions
         - error is wavelength independent
         - same error (sigma) for all atoms
        """
        n = float(tf.size(spread_posterior.sites))
        return cls(
            spread_posterior.rac, spread_posterior.Fc, tf.math.sqrt(n) * sigma)

    def flat_distribution(self):
        loc = self.loc * tf.ones(self.rac.asu_size)
        scale = self.scale * tf.ones(self.rac.asu_size)
        return tfd.Normal(loc, scale)

    def distribution(self, asu_id, hkl):
        scale = self.sigma * tf.ones_like(hkl[...,0], dtype='float32')

        # Rician RV params nu,sigma
        fc = self.rac.gather(self.Fc, asu_id, hkl)
        nu = tf.math.abs(fc)
        q = Rice(nu, self.sigma)
        return q


