import torch


class PriorBase(torch.nn.Module):
    def __init__(self, rac):
        super().__init__()
        self.rac = rac

    def distribution(self, asu_id, hkl):
        raise NotImplementedError(
            "Derived classes must implement distribution(asu_id, hkl) -> Distribution")

    def flat_distribution(self):
        raise NotImplementedError(
            "Derived classes must implement flat_distribution() -> Distribution")

    def forward(self, inputs, flat):
        if flat:
            return self.flat_distribution()
        return self.distribution(inputs.asu_id, inputs.hkl)
