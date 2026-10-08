from torch import distributions as td
from abismal import serialization
from abismal.likelihood.location_scale import LocationScale

@serialization.register
class StudentTLikelihood(LocationScale):
    def __init__(self, degrees_of_freedom):
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom

    def get_config(self):
        return {'degrees_of_freedom' : self.degrees_of_freedom}

    def _likelihood(self, iobs, sigiobs):
        return td.StudentT(self.degrees_of_freedom, iobs, sigiobs)
