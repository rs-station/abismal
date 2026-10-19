import torch


class Shifted(torch.distributions.Distribution):
    """
    `base` shifted right by a constant, ``X + low``.

    torch's TransformedDistribution would do, except that it leaves mean and
    variance unimplemented, and the posteriors are read out through exactly
    those. KL divergence is deliberately not registered for this class, so
    callers fall back on a Monte Carlo estimate as they did under TFP.
    """
    has_rsample = True
    arg_constraints = {}

    def __init__(self, base, low):
        self.base = base
        self.low = low
        super().__init__(base.batch_shape, base.event_shape, validate_args=False)

    @property
    def support(self):
        return self.base.support

    def rsample(self, sample_shape=torch.Size()):
        return self.base.rsample(sample_shape) + self.low

    def log_prob(self, value):
        return self.base.log_prob(value - self.low)

    @property
    def mean(self):
        return self.base.mean + self.low

    @property
    def variance(self):
        return self.base.variance

    @property
    def stddev(self):
        return self.base.stddev
