import numpy as np
import torch
from abismal import tracking


class RunningMoments:
    def __init__(self, axis=-2):
        self.n=0
        self.s=0
        self.mean=0
        self.axis=axis

    def update(self, x, weights=None):
        if weights is None:
            weights = np.ones_like(x)
        self.n += np.sum(weights, axis=self.axis, keepdims=True)
        diff = x - self.mean
        self.mean += np.sum(weights * diff / self.n, axis=self.axis, keepdims=True)
        self.s = self.s + np.sum(weights * diff * (x - self.mean), axis=self.axis, keepdims=True)

    @property
    def var(self):
        if np.sum(self.n) <= 1:
            return None
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt(self.var)
class Standardize(torch.nn.Module):
    """
    Standardize the last axis of its input with running (Welford) moments that
    accumulate while training and are frozen thereafter.

    The moments are buffers: they are saved with the model and moved with it,
    but no optimizer touches them.
    """
    def __init__(self, units, center=True, max_counts=np.inf, epsilon=1e-6):
        super().__init__()
        self.units = units
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts
        self.register_buffer('mean', torch.zeros(units))
        self.register_buffer('m2', torch.zeros(units))
        self.register_buffer('count', torch.zeros(()))
        # A tensor rather than a Python bool so that StandardizationFreezer can
        # flip it mid-training without recompiling a compiled train step.
        self.register_buffer('frozen', torch.zeros((), dtype=torch.bool))

    @property
    def flat_mean(self):
        return self.mean

    @property
    def flat_std(self):
        return self.std

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def var(self):
        # Before any data the moments say nothing, so standardize by mean 0 and
        # variance 1 -- the identity -- rather than divide by count = 0, which made
        # every input exactly zero and an untrained model's output meaningless
        # (NaN after an RMS-normalized layer). Training never sees this: the first
        # forward pass updates the moments before using them.
        return torch.where(
            self.count > 0,
            self.m2.clamp_min(self.epsilon) / self.count,
            torch.ones_like(self.m2),
        )

    @torch.no_grad()
    def update(self, x):
        """Fold the rows of `x`, shape (n, units), into the running moments."""
        x = x.reshape(-1, self.units)
        k = x.shape[0]
        # Gate every write on `frozen` instead of branching in Python, which
        # would be read once when a compiled step is traced and never again.
        live = ~self.frozen
        count = torch.where(live, self.count + k, self.count)
        diff = x - self.mean
        new_mean = self.mean + diff.sum(0) / count
        m2 = self.m2 + (diff * (x - new_mean)).sum(0)
        self.count.copy_(count)
        self.mean.copy_(torch.where(live, new_mean, self.mean))
        self.m2.copy_(torch.where(live, m2, self.m2))

    def standardize(self, data):
        if self.center:
            return (data - self.flat_mean) / self.flat_std
        return data / self.flat_std

    def forward(self, data, training=True):
        if training:
            self.update(data)
        return self.standardize(data)
class StandardizeMetadata(Standardize):
    def __init__(self, units, max_counts=np.inf, epsilon=1e-6, **kwargs):
        kwargs.pop('center', None)
        super().__init__(units, center=True, max_counts=max_counts, epsilon=epsilon)

    def forward(self, inputs, training=None):
        metadata_out = super().forward(inputs.metadata, training=training)
        return inputs._replace(metadata=metadata_out)
class StandardizeIntensities(Standardize):
    def __init__(self, units=1, max_counts=np.inf, epsilon=1e-6, **kwargs):
        kwargs.pop('center', None)
        super().__init__(units, center=False, max_counts=max_counts, epsilon=epsilon)

    def forward(self, inputs, training=None):
        iout = super().forward(inputs.iobs, training=training)
        sigout = self.standardize(inputs.sigiobs)
        tracking.add_metric(self.mean.squeeze(), "Imean")
        tracking.add_metric(self.std.squeeze(), "Istd")
        return inputs._replace(iobs=iout, sigiobs=sigout)
