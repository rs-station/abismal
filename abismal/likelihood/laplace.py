from torch import distributions as td
from abismal.likelihood.location_scale import LocationScale
class LaplaceLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        return td.Laplace(iobs, sigiobs)
