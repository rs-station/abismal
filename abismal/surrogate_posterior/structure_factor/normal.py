import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase,MultivariateNormalPosteriorBase
import reciprocalspaceship as rs


@tfk.saving.register_keras_serializable(package="abismal")
class NormalPosterior(NormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a normal distribution
    """


@tfk.saving.register_keras_serializable(package="abismal")
class MultivariateNormalPosterior(MultivariateNormalPosteriorBase, StructureFactorPosteriorBase):
    """
    A structure factor surrogate posterior parameterized by a low-rank multivariate normal distribution
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_flat_extras(self):
        extras = {}
        for i,v in enumerate(self.scale_update.numpy().T):
            extras[f'Comp{i}'] = rs.DataSeries(v, dtype='F')
        return extras

