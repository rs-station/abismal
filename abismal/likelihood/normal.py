import torch
from torch import distributions as td
from abismal import serialization
from abismal.likelihood.location_scale import LocationScale


@serialization.register
class WeightedLeastSquaresLikelihood(LocationScale):
    def forward(self, ipred, iobs, sigiobs):
        return torch.square(
            (ipred - iobs) / sigiobs
        )

@serialization.register
class NormalLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        return td.Normal(iobs, sigiobs)

@serialization.register
class LeastSquaresLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        scale = sigiobs.mean()
        return td.Normal(iobs, scale)
