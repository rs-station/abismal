import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from abismal.distributions import FoldedNormal,Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection
from abismal.prior.base import PriorBase


def centric_wilson(epsilon, sigma=1.):
    return tfd.HalfNormal(tf.math.sqrt(epsilon * sigma))

def acentric_wilson(epsilon, sigma=1.):
    return tfd.Weibull(2., tf.math.sqrt(epsilon * sigma))

class WilsonDistribution:
    def __init__(self, centric, epsilon, sigma=1.):
        self.p_centric = centric_wilson(epsilon, sigma)
        self.p_acentric = acentric_wilson(epsilon, sigma)
        self.centric = centric

    def mean(self):
        return tf.where(
            self.centric,
            self.p_centric.mean(),
            self.p_acentric.mean(),
        )

    def stddev(self):
        return tf.where(
            self.centric,
            self.p_centric.stddev(),
            self.p_acentric.stddev(),
        )

    def log_prob(self, z):
        ll = tf.where(
            self.centric,
            self.p_centric.log_prob(z),
            self.p_acentric.log_prob(z),
        )
        return ll

@tfk.saving.register_keras_serializable(package="abismal")
class WilsonPrior(PriorBase):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, rac, sigma=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. If this is an array it must be length rac.asu_size
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.sigma = sigma
        self.built = True #This is always true

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'sigma' : self.sigma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)

    def distribution(self, asu_id, hkl):
        centric = self.rac.gather(self.rac.centric, asu_id, hkl)
        epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
        sigma = self.sigma
        if len(tf.shape(sigma)) > 0:
            sigma = self.rac.gather(sigma, asu_id, hkl)
        p = WilsonDistribution(centric, epsilon, sigma)
        return p

@tfk.saving.register_keras_serializable(package="abismal")
class MultiWilsonPrior(tfk.layers.Layer):
    """
    This class uses reparameterized samples to approximate the log probability 
    of a multivariate Wilson prior. For this object, the user needs to specify
    a ReciprocalASUCollection instance which enumerates the sets of merged
    structure factors which will be produced. Additionally, a list of 
    parent asu identifiers will be provided. Each ASU can have at most one
    parent. The strength of the relationship between an ASU and its parent
    is given as a prior correlation coefficient. 

    Both the Single and Multi-Wilson prior can be expressed in terms of 
    folded normal and Ricean distributions. This is an important detail 
    that simplifies the implementation. 

    ```
    Wilson(F_h) = 
        FoldedNormal(0, sqrt(epsilon_h * Sigma_h)) #centric
        Rice(0., sqrt(0.5 * epsilon_h * Sigma_h)   #acentric
    ```
    where epsilon is the multiplicity and Sigma is the average reflection
    intensity. 

    ```
    DoubleWilson(F_h) = 
        FoldedNormal(r_h * z_Pa(h), sqrt(epsilon_h * Sigma_h * (1 - r^2))) #centric
        Rice(r_h * z_Pa(h), sqrt(0.5 * epsilon_h * Sigma_h * (1 - r^2)))   #acentric
    ```

    """
    def __init__(self, rac, parents, correlations, reindexing_ops=None, sigma=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
            The reciprocal asu collection describing the merged structure factors
        parents : list
            An iterable of zero-indexed parents for each asu in the rac. Use None
            to indicate root nodes.
        correlations : list
            An iterable of prior correlation coefficients between asus. Use
            0.0 for root nodes. 
        reindexing_ops : list (optional)
            Optionally provide a list of reindexing operator strings, one per asu. 
        sigma : float or tensor (optional)
            Optionally provide an average intensity value for the prior. 
            If this is a tensor, it should have the combinded length of all
            the asus in the rac. 
        """
        super().__init__(**kwargs)
        #Store these for config purposes
        self.rac = rac
        self._parents = parents
        self._correlations = correlations
        self._reindexing_ops = reindexing_ops
        self._sigma = sigma

        self.centric = rac.centric
        self.epsilon = rac.epsilon

        parent_ids = []
        is_root = []

        for asu_id, rasu in enumerate(rac):
            pa = parents[asu_id]
            if pa == asu_id or pa < 0:
                pa = None
            if pa is None:
                r = 0.
            else:
                r = correlations[asu_id]

            op = 'x,y,z'
            if reindexing_ops is not None:
                op = reindexing_ops[asu_id]
            if not isinstance(op, Op):
                op = Op(op)

            hkl = rasu.Hunique
            hkl = op(hkl)

            if pa is None:
                parent_id = -tf.ones(rasu.asu_size, rasu.miller_id.dtype)
                is_root.append(
                    tf.ones(rasu.asu_size, dtype='bool')
                )
            else:
                parent_id = rac._miller_ids(
                    pa * tf.ones_like(hkl[:,:1]),
                    hkl,
                )
                is_root.append(
                    tf.zeros(rasu.asu_size, dtype='bool')
                )
            parent_ids.append(parent_id)

        self.is_root = tf.concat(is_root, axis=0)
        self.parent_ids = tf.concat(parent_ids, axis=0)
        self.sigma = sigma

        idx = self.parent_ids >= 0
        r = tf.gather(tf.convert_to_tensor(correlations), rac.asu_id)
        self.r = tf.where(idx, r, 0.)

        self.scale = tf.where(
            rac.centric,
            tf.sqrt(rac.epsilon * sigma * (1. - tf.square(self.r))),
            tf.sqrt(0.5 * rac.epsilon * sigma * (1. - tf.square(self.r))),
        )
        self.has_parent = self.parent_ids >= 0
        self.parent_ids = tf.where(self.has_parent, self.parent_ids, tf.range(rac.asu_size, dtype=tf.int32))
        self.built = True #This is always true

        self.p_centric  = centric_wilson(self.epsilon, sigma)
        self.p_acentric = acentric_wilson(self.epsilon, sigma)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'parents' : self._parents,
            'correlations' : self._correlations,
            'reindexing_ops' : self._reindexing_ops,
            'sigma' : self.sigma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)


    def mean(self):
        """
        This is only for initialization of the surrogate!
        """
        return WilsonPrior(self.rac, self.sigma).mean()

    def log_prob(self, z):
        scale = self.scale #This is precomputed
        loc = self.r * tf.gather(z, self.parent_ids, axis=-1)

        # Mask any root nodes or missing parents
        loc = tf.where(self.has_parent, loc, 0.)

        ll  = tf.where(
            self.centric,
            FoldedNormal(loc, scale).log_prob(z),
            Rice(loc, scale).log_prob(z),
        )
        wilson_p = tf.where(
            self.centric,
            self.p_centric.log_prob(z),
            self.p_acentric.log_prob(z),
        )

        ll = tf.where(
            self.is_root,
            wilson_p,
            ll,
        )

        return ll


