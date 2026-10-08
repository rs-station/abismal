import torch
from abismal import serialization
from abismal import tracking

def weighted_pearsonr(x, y, w=None, axis=-1, keepdims=False, eps=1e-12):
    """
    Calculate a [weighted Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient).

    Note
    ----
    x, y, and w may have arbitrarily shaped leading dimensions. The correlation coefficient will always be computed pairwise along the last axis.

    Parameters
    ----------
    x : torch.Tensor
        An array of observations.
    y : torch.Tensor
        An array of observations the same shape as x.
    w : torch.Tensor
        An array of weights the same shape as x. These needn't be normalized.
    axis : int or tuple
        The axis or axes to reduce over.

    Returns
    -------
    r : torch.Tensor
        The Pearson correlation coefficient along the reduced axes.
    """
    if w is None:
        w = torch.ones_like(x)

    z = torch.reciprocal(w.sum(axis, keepdim=True))
    mx = (z * (w * x)).sum(axis, keepdim=True)
    my = (z * (w * y)).sum(axis, keepdim=True)

    dx = x - mx
    dy = y - my

    cxy = z * (w * dx * dy).sum(axis, keepdim=True)
    cx = z * (w * dx * dx).sum(axis, keepdim=True)
    cy = z * (w * dy * dy).sum(axis, keepdim=True)

    r = cxy / torch.sqrt(cx * cy + eps)
    if not keepdims:
        r = r.squeeze()
    return r

class LocationScale(torch.nn.Module, serialization.Serializable):
    def get_config(self):
        return {}

    def _likelihood(self, iobs, sigiobs):
        raise NotImplementedError(
            "Derived classes must implement _likelihood(iobs, sigiobs)->torch.distributions.Distribution"
        )

    @torch.no_grad()
    def register_metrics(self, ipred, iobs, sigiobs):
        # ipred is (reflections, mc samples) and iobs, sigiobs are (reflections, 1)
        iobs = iobs * torch.ones_like(ipred)
        sigiobs = sigiobs * torch.ones_like(ipred)
        w = torch.reciprocal(torch.square(sigiobs))

        cc = weighted_pearsonr(iobs, ipred, w, axis=(-2, -1))
        tracking.add_metric(cc, 'wCCpred')

        cc = weighted_pearsonr(iobs, ipred, axis=(-2, -1))
        tracking.add_metric(cc, 'CCpred')

    def forward(self, ipred, iobs, sigiobs):
        likelihood = self._likelihood(iobs, sigiobs)
        ll = likelihood.log_prob(ipred)
        return ll
