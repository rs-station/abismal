import numpy as np
import tensorflow as tf
from abismal.distributions import FoldedNormal,Nakagami,Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection
from abismal.prior.base import PriorBase
from abismal.prior.wilson import WilsonPriorBase


# Wilson's distributions on structure factor amplitudes are both Nakagami:
#   centric:  Nakagami(m=1/2, Omega=eps*Sigma) = HalfNormal(scale=sqrt(eps*Sigma))
#   acentric: Nakagami(m=1,   Omega=eps*Sigma) = Weibull(2, sqrt(eps*Sigma))

def centric_wilson(epsilon, sigma=1.):
    return Nakagami(0.5, epsilon * sigma)

def acentric_wilson(epsilon, sigma=1.):
    return Nakagami(1.0, epsilon * sigma)

def wilson_distribution(centric, epsilon, sigma=1.):
    """Single Nakagami distribution covering both centric and acentric Wilson priors."""
    omega = tf.convert_to_tensor(epsilon * sigma)
    m = tf.where(
        centric,
        tf.constant(0.5, dtype=omega.dtype),
        tf.constant(1.0, dtype=omega.dtype),
    )
    return Nakagami(m, omega)


class WilsonDistribution:
    """Thin wrapper kept for backward compatibility.

    Preserves the ``mean/stddev/log_prob`` interface used by
    ``MultiWilsonDistribution``; the underlying object is a single
    ``Nakagami`` whose concentration is chosen per reflection.
    """
    def __init__(self, centric, epsilon, sigma=1.):
        self._dist = wilson_distribution(centric, epsilon, sigma)

    def mean(self):
        return self._dist.mean()

    def stddev(self):
        return self._dist.stddev()

    def log_prob(self, z):
        return self._dist.log_prob(z)

@tfk.saving.register_keras_serializable(package="abismal")
class WilsonPrior(WilsonPriorBase):
    """Wilson's priors on structure factor amplitudes."""

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
        # Return a real tfd.Distribution (Nakagami) so that KL against a
        # Nakagami posterior dispatches to the registered analytical form.
        p = wilson_distribution(centric, epsilon, sigma)
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

    def log_prob(self, z, z_pa=None):
        if z_pa is None:
            z_h = z
            z_pa = tf.gather(z, self.parent_id, axis=-1)
        else:
            z_h, z_pa = z[...,0],z[...,1]

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
