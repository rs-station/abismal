import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from abismal.likelihood.location_scale import LocationScale

class StudentTLikelihood(LocationScale):
    def __init__(self, degrees_of_freedom):
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom

    def _likelihood(self, iobs, sigiobs):
        return  tfd.StudentT(self.degrees_of_freedom, iobs, sigiobs)

