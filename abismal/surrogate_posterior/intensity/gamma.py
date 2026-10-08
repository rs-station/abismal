from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.gamma import GammaPosteriorBase
import torch
from abismal import serialization
from abismal.distributions import log_gamma_half_ratio_excess


@serialization.register
class GammaPosterior(GammaPosteriorBase, IntensityPosteriorBase):
    """
    an intensity surrogate posterior parameterized by a gamma distribution.
    """
    def get_flat_fsigf(self):
        """
        the square root of a gamma distribution is the nakagami distribution. 
        compute the mean and standard deviation of the square root of a gamma distribution.
        see [https://en.wikipedia.org/wiki/nakagami_distribution#random_variate_generation]. 
        """
        q = self.flat_distribution()
        conc = q.concentration
        rate  = q.rate

        # r = lgamma(conc + 1/2) - lgamma(conc) - log(conc)/2, which tends to 0
        # as conc grows. Writing the moments in terms of r avoids computing the
        # variance as the difference of two nearly equal numbers, conc/rate and
        # mean**2, which loses every digit for a well determined reflection.
        r = log_gamma_half_ratio_excess(conc)
        mean = torch.exp(r) * torch.sqrt(conc / rate)
        var = -torch.expm1(2. * r) * conc / rate
        var = torch.where(var <= 0., self.epsilon * self.epsilon, var)
        std = torch.sqrt(var)
        return mean,std
