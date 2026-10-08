import torch
from abismal.distributions import TruncatedNormal as _TruncatedNormal
from abismal.surrogate_posterior.folded_normal import FoldedNormalPosteriorBase

class TruncatedNormal(_TruncatedNormal):
    def rsample(self, *args, **kwargs):
        z = super().rsample(*args, **kwargs)
        return torch.maximum(z, torch.as_tensor(self.low, dtype=z.dtype, device=z.device))

class TruncatedNormalPosteriorBase(FoldedNormalPosteriorBase):
    """
    A base class for creating truncated normal posteriors. 
    """
    high = 1e10

    def _distribution(self, loc, scale, low):
        q = TruncatedNormal(
            loc + low, 
            scale, 
            low=low,
            high=self.high,
        )
        return q
