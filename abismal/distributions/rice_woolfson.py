import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

from abismal.distributions.folded_normal import FoldedNormal
from abismal.distributions.rice import Rice


class RiceWoolfson(Distribution):
    """Folded normal where ``centric`` is true and Rice elsewhere.

    Both component distributions are evaluated everywhere and selected with
    ``torch.where``, which keeps shapes static; their gradients are finite in the
    discarded branch, so the selection does not leak NaNs.
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, loc, scale, centric, validate_args=False):
        self.loc, self.scale = broadcast_all(loc, scale)
        centric = torch.as_tensor(centric, dtype=torch.bool, device=self.loc.device)
        batch_shape = torch.broadcast_shapes(self.loc.shape, centric.shape)
        self.loc = self.loc.expand(batch_shape)
        self.scale = self.scale.expand(batch_shape)
        self.centric = centric.expand(batch_shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def _centric_distribution(self):
        return FoldedNormal(self.loc, self.scale, validate_args=self._validate_args)

    @property
    def _acentric_distribution(self):
        return Rice(self.loc, self.scale, validate_args=self._validate_args)

    def _select(self, centric_value, acentric_value):
        return torch.where(self.centric, centric_value, acentric_value)

    def log_prob(self, value):
        return self._select(
            self._centric_distribution.log_prob(value),
            self._acentric_distribution.log_prob(value),
        )

    def rsample(self, sample_shape=torch.Size()):
        return self._select(
            self._centric_distribution.rsample(sample_shape),
            self._acentric_distribution.rsample(sample_shape),
        )

    @property
    def mean(self):
        return self._select(
            self._centric_distribution.mean, self._acentric_distribution.mean
        )

    @property
    def variance(self):
        return self._select(
            self._centric_distribution.variance, self._acentric_distribution.variance
        )

    @property
    def stddev(self):
        return self._select(
            self._centric_distribution.stddev, self._acentric_distribution.stddev
        )
