import torch
from abismal import serialization


class PriorBase(torch.nn.Module, serialization.Serializable):
    def __init__(self, rac):
        super().__init__()
        self.rac = rac

    def get_config(self):
        return {'rac' : serialization.serialize(self.rac)}

    @classmethod
    def from_config(cls, config):
        config['rac'] = serialization.deserialize(config['rac'])
        return cls(**config)

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
