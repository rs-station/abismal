import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.surrogate_posterior import IntensityPosteriorBase
from abismal.surrogate_posterior.normal import NormalPosteriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class NormalPosterior(NormalPosteriorBase, IntensityPosteriorBase):
    """
    An intensity surrogate posterior parameterized by a folded normal distribution
    """
