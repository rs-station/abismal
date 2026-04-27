import tf_keras as tfk

from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.nakagami import NakagamiPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class NakagamiPosterior(NakagamiPosteriorBase, StructureFactorPosteriorBase):
    """A structure factor surrogate posterior parameterized by a Nakagami distribution.

    When paired with a Wilson (Nakagami) prior, the KL divergence is analytical.
    """

    def get_flat_isigi(self):
        g = self.flat_distribution().square()
        return g.mean(), g.stddev()
