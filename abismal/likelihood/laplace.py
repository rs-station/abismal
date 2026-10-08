from torch import distributions as td
from abismal import serialization
from abismal.likelihood.location_scale import LocationScale

@serialization.register
class LaplaceLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        return td.Laplace(iobs, sigiobs)
