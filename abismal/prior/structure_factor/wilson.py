import numpy as np
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

    def _distribution(self, asu_id=None, hkl=None):
        sigma = self.sigma
        if asu_id is None:
            centric = self.rac.centric
            epsilon = self.rac.epsilon
            asu_id = self.rac.asu_id
        else:
            centric = self.rac.gather(self.rac.centric, asu_id, hkl)
            epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
            if len(tf.shape(sigma)) > 0:
                sigma = self.rac.gather(sigma, asu_id, hkl)
        p = WilsonDistribution(centric, epsilon, sigma)
        return p

    def flat_distribution(self):
        return self._distribution(asu_id=None, hkl=None)

    def distribution(self, asu_id, hkl):
        return self._distribution(asu_id, hkl)

class MultiWilsonDistribution:
    def __init__(self, is_root, correlation, centric, multiplicity, sigma=1., parent_id=None):
        self.parent_id = parent_id #Convert this to a flat distribution
        self.is_root = is_root
        self.correlation = correlation
        self.centric = centric
        self.multiplicity = multiplicity
        self.sigma = sigma

    def mean(self):
        """ 
        This is purely for initialization purposes and is not to be trusted.
        """ 
        loc = tf.where(
            self.centric,
            centric_wilson(self.multiplicity, self.sigma).mean(),
            acentric_wilson(self.multiplicity, self.sigma).mean(),
        )
        return loc

    def log_prob(self, z):
        if self.parent_id is not None:
            z_h = z
            z_pa = tf.gather(z, self.parent_id, axis=-1)
        else:
            z_h, z_pa = tf.unstack(z, axis=-1)

        #Single wilson case for root nodes
        ll_sw = WilsonDistribution(self.centric, self.multiplicity, self.sigma).log_prob(z_h)

        #Double wilson case for child nodes
        loc = self.correlation * z_pa
        scale = tf.sqrt(self.multiplicity * (1. - tf.square(self.correlation))),
        ll_dw = tf.where(
            self.centric,
            FoldedNormal(loc, scale).log_prob(z_h),
            Rice(loc, tf.sqrt(0.5) * scale).log_prob(z_h),
        )

        #Put them both together
        ll = tf.where(
            self.is_root,
            ll_sw,
            ll_dw,
        )
        return ll

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
    def __init__(self, rac, correlation, sigma=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
            The reciprocal asu collection describing the merged structure factors
        correlation : list
            An iterable of prior correlation coefficients between asus. Use
            0.0 for root nodes. 
        sigma : float or tensor (optional)
            Optionally provide an average intensity value for the prior. 
            If this is a tensor, it should have the combinded length of all
            the asus in the rac. 
        """
        super().__init__(**kwargs)
        self.rac = rac
        self._correlation = correlation
        self.sigma = sigma
        self.built = True #This is always true

    @property
    def correlation(self):
        return tf.gather(self._correlation, self.rac.asu_id)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'correlation' : self._correlation,
            'sigma' : self.sigma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        return cls(**config)

    def _distribution(self, asu_id=None, hkl=None):
        sigma = self.sigma
        if asu_id is None:
            root = self.rac.is_root
            centric = self.rac.centric
            epsilon = self.rac.epsilon
            correlation =self.correlation
            parent_id = self.rac.parent_miller_id
        else:
            root = self.rac.gather(self.rac.is_root, asu_id, hkl)
            centric = self.rac.gather(self.rac.centric, asu_id, hkl)
            epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
            correlation = tf.squeeze(tf.gather(self.correlation, asu_id), axis=-1)
            parent_id = None
            asu_id = self.rac.asu_id
            hkl = self.rac.Hunique
            if len(tf.shape(sigma)) > 0:
                sigma = tf.squeeze(
                    self.rac.gather(self.sigma, asu_id, hkl),
                    axis=-1,
                )
        p = MultiWilsonDistribution(root, correlation, centric, epsilon, sigma, parent_id=parent_id)
        return p

    def flat_distribution(self):
        return self._distribution(asu_id=None, hkl=None)

    def distribution(self, asu_id, hkl):
        return self._distribution(asu_id, hkl)
