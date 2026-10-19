import math
import torch
from abismal.distributions import FoldedNormal,Nakagami,Rice
from abismal.prior.base import PriorBase
from abismal.prior.wilson import WilsonPriorBase,AutoWilsonPriorBase


# Wilson's distributions on structure factor amplitudes are both Nakagami:
#   centric:  Nakagami(m=1/2, Omega=eps*Sigma) = HalfNormal(scale=sqrt(eps*Sigma))
#   acentric: Nakagami(m=1,   Omega=eps*Sigma) = Weibull(2, sqrt(eps*Sigma))

def centric_wilson(epsilon, sigma=1.):
    return Nakagami(0.5, epsilon * sigma)

def acentric_wilson(epsilon, sigma=1.):
    return Nakagami(1.0, epsilon * sigma)

def wilson_distribution(centric, epsilon, sigma=1.):
    """Single Nakagami distribution covering both centric and acentric Wilson priors."""
    omega = torch.as_tensor(epsilon * sigma)
    m = torch.where(centric, 0.5, 1.0).to(omega.dtype)
    return Nakagami(m, omega)


class WilsonDistribution:
    """Thin wrapper kept for backward compatibility.

    Preserves the ``mean/stddev/log_prob`` interface used by
    ``MultiWilsonDistribution``; the underlying object is a single
    ``Nakagami`` whose concentration is chosen per reflection.
    """
    def __init__(self, centric, epsilon, sigma=1.):
        self._dist = wilson_distribution(centric, epsilon, sigma)

    @property
    def mean(self):
        return self._dist.mean

    @property
    def stddev(self):
        return self._dist.stddev

    def log_prob(self, z):
        return self._dist.log_prob(z)
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
            sigma = self.rac.gather(sigma, asu_id, hkl)
        # Return a real torch Distribution (Nakagami) so that KL against a
        # Nakagami posterior dispatches to the registered analytical form.
        p = wilson_distribution(centric, epsilon, sigma)
        return p

    def flat_distribution(self):
        return self._distribution(asu_id=None, hkl=None)

    def distribution(self, asu_id, hkl):
        return self._distribution(asu_id, hkl)
class AutoWilsonPrior(WilsonPrior, AutoWilsonPriorBase):
    """Wilson prior with learnable scale and b-factor"""

class MultiWilsonDistribution:
    def __init__(self, is_root, correlation, centric, multiplicity, sigma=1., parent_id=None):
        self.parent_id = parent_id #Convert this to a flat distribution
        self.is_root = is_root
        self.correlation = correlation
        self.centric = centric
        self.multiplicity = multiplicity
        self.sigma = sigma

    @property
    def mean(self):
        """ 
        This is purely for initialization purposes and is not to be trusted.
        """ 
        loc = torch.where(
            self.centric,
            centric_wilson(self.multiplicity, self.sigma).mean,
            acentric_wilson(self.multiplicity, self.sigma).mean,
        )
        return loc

    @property
    def stddev(self):
        """Marginal standard deviation of the single-Wilson (root) distribution.

        Initialization only, like :meth:`mean` -- it deliberately ignores the
        double-Wilson coupling, so for a child node this is the prior width
        before conditioning on the parent, not the conditional width.
        """
        return torch.where(
            self.centric,
            centric_wilson(self.multiplicity, self.sigma).stddev,
            acentric_wilson(self.multiplicity, self.sigma).stddev,
        )

    def log_prob(self, z, z_pa=None):
        if z_pa is None:
            z_h = z
            z_pa = z[..., self.parent_id.clamp_min(0)]
        else:
            z_h, z_pa = z[...,0],z[...,1]

        #Single wilson case for root nodes
        ll_sw = WilsonDistribution(self.centric, self.multiplicity, self.sigma).log_prob(z_h)

        #Double wilson case for child nodes
        loc = self.correlation * z_pa
        # sqrt(epsilon_h * Sigma_h * (1 - r^2)), per the class docstring. Sigma
        # was missing here while the root branch above passes it, so child nodes
        # got a prior width off by sqrt(Sigma). Invisible at the default
        # sigma=1.0 and wrong for empirical_wilson / AutoWilsonPrior.
        scale = torch.sqrt(
            self.multiplicity * self.sigma * (1. - torch.square(self.correlation))
        )
        # Root nodes have correlation 0 and no parent, so these are evaluated
        # there only to be discarded by the where below.
        ll_dw = torch.where(
            self.centric,
            FoldedNormal(loc, scale).log_prob(z_h),
            Rice(loc, math.sqrt(0.5) * scale).log_prob(z_h),
        )

        #Put them both together
        ll = torch.where(
            self.is_root,
            ll_sw,
            ll_dw,
        )
        return ll
class MultiWilsonPrior(PriorBase):
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
        super().__init__(rac)
        self._correlation_list = [float(c) for c in correlation]
        self.register_buffer('_correlation', torch.tensor(self._correlation_list))
        sigma = torch.as_tensor(sigma, dtype=torch.float32) * torch.ones(self.rac.asu_size)
        self.register_buffer('sigma', sigma)

    @property
    def correlation(self):
        return self._correlation[self.rac.asu_id]

    def _distribution(self, asu_id=None, hkl=None):
        sigma = self.sigma
        if asu_id is None:
            root = self.rac.is_root
            centric = self.rac.centric
            epsilon = self.rac.epsilon
            correlation =self.correlation
            parent_id = self.rac.parent_miller_id
        else:
            # The double-Wilson conditional needs the PARENT reflection's
            # posterior sample, which for a batch is generally not in that
            # batch: `log_prob` gathers `z` at `parent_miller_id`, indices into
            # the full ASU collection, while a non-flat `z` runs over the
            # observations present. Indexing one with the other either raises
            # (small batch) or silently conditions each reflection on an
            # unrelated observation (large batch).
            #
            # abismal never reaches here -- command_line/abismal.py forces
            # posterior_kwargs['independent']=False for this prior, so only
            # flat_distribution() runs. Fail loudly rather than return numbers
            # that look plausible.
            raise NotImplementedError(
                "MultiWilsonPrior has no per-batch distribution: the "
                "double-Wilson conditional needs the parent reflection's "
                "posterior sample, which a batch does not generally contain. "
                "Use flat_distribution() (independent=False), which is what "
                "abismal does."
            )
            root = self.rac.gather(self.rac.is_root, asu_id, hkl)
            centric = self.rac.gather(self.rac.centric, asu_id, hkl)
            epsilon = self.rac.gather(self.rac.epsilon, asu_id, hkl)
            # `self.correlation` is the per-Miller-index expansion, so indexing
            # it by asu_id would return the correlation of Miller ids 0..n_asu,
            # silently yielding ~0 for most reflections. The per-ASU vector is
            # `self._correlation`.
            correlation = self._correlation[asu_id]
            # log_prob gathers z at these indices, so they must be flat Miller
            # ids into the full ASU collection -- the same space
            # `rac.parent_miller_id` is already in.
            parent_id = self.rac.gather(self.rac.parent_miller_id, asu_id, hkl)
            # Gather with the CALLER's asu_id/hkl. These used to be
            # overwritten with the full-ASU vectors just above, which both
            # mismatched the batch length and passed a rank-2 hkl where a
            # rank-1 index was expected.
            sigma = self.rac.gather(self.sigma, asu_id, hkl)
        p = MultiWilsonDistribution(root, correlation, centric, epsilon, sigma, parent_id=parent_id)
        return p

    def flat_distribution(self):
        return self._distribution(asu_id=None, hkl=None)

    def distribution(self, asu_id, hkl):
        return self._distribution(asu_id, hkl)
