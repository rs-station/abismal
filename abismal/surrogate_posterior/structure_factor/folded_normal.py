import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.distributions import FoldedNormal as FoldedNormal
from abismal.surrogate_posterior import StructureFactorPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class FoldedNormalPosterior(StructureFactorPosteriorBase):
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

class DoubleFoldedNormal:
    def __init__(self, loc_1, scale_1, loc_2, scale_2, low=1e-12):
        """
        **Note use scale_2 <= 0. to signify that the parent distribution is absent
        """
        self.no_parent = (scale_2 <= 0.)
        scale_2 = tf.where(self.no_parent, 1., scale_2)
        self.p_1 = FoldedNormal(loc_1, scale_1)
        self.p_2 = FoldedNormal(loc_2, scale_2)
        self.event_shape = tf.TensorShape(2)
        self.low = low

    def sample(self, *args, **kwargs):
        z_1 = self.p_1.sample(*args, **kwargs)
        z_2 = self.p_2.sample(*args, **kwargs)
        z_2 = tf.where(self.no_parent[None,:], 0., z_2)
        z = tf.concat(( 
            z_1[...,None],
            z_2[...,None],
        ), axis=-1) + self.low
        return z

    def kl_divergence(self, *args, **kwargs):
        raise NotImplementedError(f"No analytical KL div for {self}")

    def log_prob(self, z):
        ll = self.p_1.log_prob(z[...,0])
        return ll

@tfk.saving.register_keras_serializable(package="abismal")
class MultivariateFoldedNormalPosterior(StructureFactorPosteriorBase):
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
        loc_1 = self.rac.gather(self.loc, asu_id, hkl)
        scale_1 = self.rac.gather(self.scale, asu_id, hkl)

        pa = self.rac.gather(self.rac.parent_miller_id, asu_id, hkl)
        loc_2 = tf.gather(self.loc, pa)
        scale_2 = tf.gather(self.scale, pa)
        q = DoubleFoldedNormal(loc_1, scale_1, loc_2, scale_2, self.low)
        return q

        from IPython import embed
        embed(colors='linux')
        XX
        loc_1 = self.rac.gather(self.loc, asu_id, hkl)
        scale_1 = self.rac.gather(self.scale, asu_id, hkl)
        q = self._distribution(loc, scale, self.low)
        return q

    def flat_distribution(self):
        """
        This is used for output of F,SIGF and is univariate
        """
        q = self._distribution(self.loc, self.scale, self.low)
        return q

