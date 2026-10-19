import torch
from torch import distributions as td
from abismal.likelihood.location_scale import LocationScale
class WeightedLeastSquaresLikelihood(LocationScale):
    def forward(self, ipred, iobs, sigiobs):
        return torch.square(
            (ipred - iobs) / sigiobs
        )
class NormalLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        return td.Normal(iobs, sigiobs)
class LeastSquaresLikelihood(LocationScale):
    def _likelihood(self, iobs, sigiobs):
        scale = sigiobs.mean()
        return td.Normal(iobs, scale)
