import torch
from torch import distributions as td
from abismal.surrogate_posterior.surrogate_posterior import PositiveParameter



class NormalPosteriorBase(object):
    """
    A base class for creating normal posteriors. 
    """
    def __init__(self, rac, loc_init=None, scale_init=None, epsilon=1e-12, **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)

        if loc_init is None:
            loc_init = torch.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * torch.ones(rac.asu_size)

        self.loc = PositiveParameter(loc_init)
        self.scale = PositiveParameter(scale_init, shift=epsilon)

    def _distribution(self, loc, scale):
        q = td.Normal(
            loc, 
            scale, 
        )
        return q

    def distribution(self, asu_id, hkl):
        loc = self.loc.gather(self.rac, asu_id, hkl)
        scale = self.scale.gather(self.rac, asu_id, hkl)
        q = self._distribution(loc, scale)
        return q

    def flat_distribution(self):
        q = self._distribution(self.loc(), self.scale())
        return q


class MultivariateNormalPosteriorBase(object):
    """
    A base class for creating low-rank multivariate normal posteriors. 
    """
    def __init__(self, rac, rank, loc_init=None, independent=False, scale_init=None, epsilon=1e-12, **kwargs):
        super().__init__(rac, epsilon=epsilon, independent=independent, **kwargs)
        if loc_init is None:
            loc_init = torch.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * torch.ones(rac.asu_size)

        self.loc = PositiveParameter(loc_init)

        self._scale = PositiveParameter(
            torch.as_tensor(scale_init, dtype=torch.float32).mean(),
            shift=epsilon,
        )
        self.rank = rank
        self.scale_update = torch.nn.Parameter(torch.zeros(rac.asu_size, rank))

    @property
    def scale(self):
        return torch.ones(self.rac.asu_size, device=self._scale.raw.device) * self._scale()

    def get_config(self):
        conf = super().get_config()
        conf.update({
            'rank' : self.rank,
        })
        return conf

    def _distribution(self, loc, scale, scale_update):
        # TFP's MultivariateNormalDiagPlusLowRankCovariance(loc, d, U) is the
        # normal with covariance diag(d) + U U^T, which is exactly torch's
        # LowRankMultivariateNormal(loc, cov_factor=U, cov_diag=d).
        q = td.LowRankMultivariateNormal(
            loc, 
            scale_update,
            scale, 
        )
        return q

    def distribution(self, asu_id, hkl):
        loc = self.loc.gather(self.rac, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        scale_update = self.rac.gather(self.scale_update, asu_id, hkl)
        q = self._distribution(loc, scale, scale_update)
        return q

    def flat_distribution(self):
        q = self._distribution(self.loc(), self.scale, self.scale_update)
        return q
