import tf_keras as tfk

from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.lognormal import LogNormalPosteriorBase


@tfk.saving.register_keras_serializable(
    package="abismal", name="IntensityLogNormalPosterior"
)
class LogNormalPosterior(LogNormalPosteriorBase, IntensityPosteriorBase):
    """An intensity surrogate posterior parameterized by a log normal distribution."""

    def get_flat_fsigf(self):
        """Exact, rather than the propagated estimate the base class falls back on.

        The square root of a log normal is log normal: it halves the underlying
        normal, so F ~ LogNormal(loc / 2, scale / 2) and both of its moments are
        closed form.
        """
        q = self.flat_distribution()
        f = self._distribution(0.5 * q.loc, 0.5 * q.scale)
        return f.mean(), f.stddev()
