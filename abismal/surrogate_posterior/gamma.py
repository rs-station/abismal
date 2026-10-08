import torch
from torch import distributions as td
from abismal.surrogate_posterior.surrogate_posterior import PositiveParameter



class GammaPosteriorBase(object):
    """
    A base class for creating gamma posteriors. 
    """
    def __init__(self, rac, loc_init=None, scale_init=None,  epsilon=1e-12, concentration_min=0., **kwargs):
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.concentration_min = concentration_min

        if loc_init is None:
            loc_init = torch.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * loc_init

        loc_init = torch.as_tensor(loc_init, dtype=torch.float32)
        scale_init = torch.as_tensor(scale_init, dtype=torch.float32)
        conc_init = torch.square(loc_init / scale_init)
        rate_init = conc_init / loc_init

        self.rate = PositiveParameter(rate_init, shift=epsilon)

        #Concentration should remain above one to prevent change in curvature
        self.concentration = PositiveParameter(conc_init, shift=concentration_min + epsilon)

    def get_config(self):
        config = super().get_config()
        config['concentration_min'] = self.concentration_min
        return config

    def _distribution(self, concentration, rate):
        return td.Gamma(concentration, rate)

    def distribution(self, asu_id, hkl):
        rate = self.rate.gather(self.rac, asu_id, hkl)
        concentration = self.concentration.gather(self.rac, asu_id, hkl)
        q = self._distribution(concentration, rate)
        return q

    def flat_distribution(self):
        q = self._distribution(self.concentration(), self.rate())
        return q

