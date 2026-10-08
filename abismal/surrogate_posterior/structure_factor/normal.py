from abismal import serialization
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase,MultivariateNormalPosteriorBase
import reciprocalspaceship as rs


@serialization.register
class NormalPosterior(NormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a normal distribution
    """


@serialization.register
class MultivariateNormalPosterior(MultivariateNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a low-rank multivariate normal distribution
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_flat_extras(self):
        extras = {}
        for i,v in enumerate(self.scale_update.detach().cpu().numpy().T):
            extras[f'Comp{i}'] = rs.DataSeries(v, dtype='F')
        return extras

