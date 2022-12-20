import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from IPython import embed

from abismal_zero.layers import *
from abismal_zero.blocks import *

class TruncatedNormalLayer(tfk.layers.Layer):
    def __init__(self, zero=0., infinity=1e32, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.zero = zero
        self.infinity = infinity
        self.scale_bijector = tfb.Chain([
            tfb.Shift(eps),
            tfb.Softplus(),
        ])
        self.loc_bijector = tfb.Softplus()

    def call(self, inputs):
        mu,sigma = tf.unstack(inputs, axis=-1)
        sigma = self.scale_bijector(sigma)
        mu = self.loc_bijector(mu)
        q = tfd.TruncatedNormal(mu, sigma, self.zero, self.infinity)
        return q

class LocationScaleLayer(tfk.layers.Layer):
    def __init__(self, eps=1e-6, scale_bijector=None, loc_bijector=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        if scale_bijector is None:
            self.scale_bijector = tfb.Chain([
                tfb.Shift(eps),
                tfb.Softplus(),
            ])
        else:
            self.scale_bijector = scale_bijector
        self.loc_bijector = loc_bijector

    def _input_to_params(self, inputs):
        loc,scale = tf.unstack(inputs, axis=-1)
        if self.scale_bijector is not None:
            scale = self.scale_bijector(scale)
        if self.loc_bijector is not None:
            loc = self.loc_bijector(loc)
        return loc,scale 

class NormalLayer(LocationScaleLayer):
    def call(self, inputs):
        loc,scale = self._input_to_params(inputs)
        q = tfd.Normal(loc, scale)
        return q

class LogNormalLayer(LocationScaleLayer):
    def call(self, inputs):
        loc,scale = self._input_to_params(inputs)
        q = tfd.LogNormal(loc, scale)
        return q

