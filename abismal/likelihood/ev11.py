import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb

from abismal.likelihood.location_scale import LocationScale

class EV11Likelihood(LocationScale):
    def __init__(self):
        super().__init__()

    def build(self, shapes):
        self.a = tfu.TransformedVariable(
            1.0,
            tfb.Exp(),
        )
        self.b = tfu.TransformedVariable(
            1e-8,
            tfb.Exp(),
        )
        self.c = tfu.TransformedVariable(
            1.,
            tfb.Exp(),
        )

    def _likelihood(self, iobs, sigiobs, imodel, scale):
        #imodel = tf.stop_gradient(imodel)
        sigma = sigiobs
        mean_imodel = tf.math.reduce_mean(imodel, axis=-1, keepdims=True)
        mean_scale = tf.math.reduce_mean(scale, axis=-1, keepdims=True)
        sigma = self.a * tf.math.sqrt(
            tf.math.square(sigma) + \
            self.b * tf.math.square(mean_scale) + \
            self.c * tf.math.square(mean_imodel)
        )
        self.add_metric(self.a, 'EV11a')
        self.add_metric(self.b, 'EV11b')
        self.add_metric(self.c, 'EV11c')
        return  tfd.Normal(iobs, sigma)

