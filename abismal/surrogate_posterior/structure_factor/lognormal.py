import tf_keras as tfk

from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.lognormal import LogNormalPosteriorBase


@tfk.saving.register_keras_serializable(
    package="abismal", name="StructureFactorLogNormalPosterior"
)
class LogNormalPosterior(LogNormalPosteriorBase, StructureFactorPosteriorBase):
    """A structure factor surrogate posterior parameterized by a log normal distribution."""

    def get_flat_isigi(self):
        """Exact, rather than the propagated estimate the base class falls back on.

        If F is log normal then so is F**2: squaring doubles the underlying
        normal, so I ~ LogNormal(2 loc, 2 scale) and both of its moments are
        closed form.
        """
        q = self.flat_distribution()
        i = self._distribution(2.0 * q.loc, 2.0 * q.scale)
        return i.mean(), i.stddev()
