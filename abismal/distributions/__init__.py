from .folded_normal import FoldedNormal
from .nakagami import Nakagami
from .rice import Rice
from .truncated_normal import TruncatedNormal
from .special import log_gamma_half_ratio, log_gamma_half_ratio_excess

# torch checks every distribution's parameters on construction unless told not
# to. TFP never did, and the check costs a GPU synchronization each time and
# breaks torch.compile graphs. It also rejects what abismal relies on TFP's
# behavior for: evaluating a prior's log density at a Monte Carlo sample that
# falls outside its support (a normal scale posterior against an exponential
# prior, say). Every abismal model path imports this package, so set it here.
import torch as _torch
_torch.distributions.Distribution.set_default_validate_args(False)
