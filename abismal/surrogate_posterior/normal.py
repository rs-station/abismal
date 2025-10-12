import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase



class NormalPosteriorBase(object):
    """
    A base class for creating normal posteriors. 
    """
    def __init__(self, rac, loc_init=None, scale_init=None, epsilon=1e-12, **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)

        if loc_init is None:
            loc_init = tf.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * tf.ones(rac.asu_size)

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

    def _distribution(self, loc, scale):
        q = tfd.Normal(
            loc, 
            scale, 
        )
        return q

    def distribution(self, asu_id, hkl):
        loc = self.rac.gather(self.loc, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        q = self._distribution(loc, scale)
        return q

    def flat_distribution(self):
        q = self._distribution(self.loc, self.scale)
        return q


class MultivariateNormalPosteriorBase(object):
    """
    A base class for creating low-rank multivariate normal posteriors. 
    """
    def __init__(self, rac, rank, loc_init=None, scale_init=None, epsilon=1e-12, **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)

        if loc_init is None:
            loc_init = tf.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * tf.ones(rac.asu_size)

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
        self.rank = rank
        self.scale_update = self.add_weight(
            shape=(rac.asu_size, rank),
            initializer="zeros",
            trainable=True,
            name="scale_update",
        )

        self.built = True

    def get_config(self):
        conf = super().get_config()
        conf.update({
            'rank' : self.rank,
        })
        return conf

    def _distribution(self, loc, scale, scale_update):
        q = tfd.MultivariateNormalDiagPlusLowRankCovariance(
            loc, 
            scale, 
            scale_update,
        )
        return q

    def distribution(self, asu_id, hkl):
        loc = self.rac.gather(self.loc, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        scale_update = self.rac.gather(self.scale_update, asu_id, hkl)
        q = self._distribution(loc, scale, scale_udpate)
        return q

    def flat_distribution(self):
        q = self._distribution(self.loc, self.scale, self.scale_update)
        return q

