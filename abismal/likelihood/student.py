import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from abismal.likelihood.location_scale import LocationScale

class StudentTLikelihood(LocationScale):
    def __init__(self, degrees_of_freedom):
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom

    def _likelihood(self, iobs, sigiobs, *args, **kwargs):
        loc = iobs
        if self.degrees_of_freedom > 2.:
            scale = sigiobs * tf.math.sqrt((self.degrees_of_freedom - 2.) / self.degrees_of_freedom)
        else:
            scale = sigiobs 
        return  tfd.StudentT(self.degrees_of_freedom, loc, scale)

class AdaptiveStudentTLikelihood(LocationScale):
    def __init__(self, degrees_of_freedom, epsilon=1e-6):
        super().__init__()
        self.init_degrees_of_freedom = degrees_of_freedom
        self.epsilon = epsilon

    def build(self, shapes):
        self.degrees_of_freedom = tfu.TransformedVariable(
            self.init_degrees_of_freedom,
            tfb.Chain((
                tfb.Shift(2.0 + self.epsilon),
                tfb.Exp(),
            )),
        )

    def _likelihood(self, iobs, sigiobs, *args, **kwargs):
        self.add_metric(self.degrees_of_freedom, name='ν')
        loc = iobs
        if self.degrees_of_freedom > 2.:
            scale = sigiobs * tf.math.sqrt((self.degrees_of_freedom - 2.) / self.degrees_of_freedom)
        else:
            scale = sigiobs 
        return  tfd.StudentT(self.degrees_of_freedom, loc, scale)
