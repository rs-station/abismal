import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection


class PriorBase(tfk.layers.Layer):
    def distribution(self, asu_id, hkl):
        raise NotImplementedError(
            "Derived classes must implement distribution(asu_id, hkl) -> Distribution")
