from abismal import serialization

from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.nakagami import NakagamiPosteriorBase


@serialization.register
class NakagamiPosterior(NakagamiPosteriorBase, StructureFactorPosteriorBase):
    """A structure factor surrogate posterior parameterized by a Nakagami distribution.

    When paired with a Wilson (Nakagami) prior, the KL divergence is analytical.
    """

    def get_flat_isigi(self):
        g = self.flat_distribution().square()
        return g.mean, g.stddev
