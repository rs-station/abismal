import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.surrogate_posterior import StructureFactorPosteriorBase



class SpreadPosterior(object):
    def __init__(self, rac, mlp, Fc, sites, wavelength_range, epsilon=1e-12, **kwargs):
        """
        rac : ReciprocalASUCollection
        mlp : tfk.layers>Layer
        Fc : np.array
        sites : np.array (n, 3)
        wavelength_range : list[float] (2)
        epsilon : float
        """
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.Fc = Fc
        self.sites = sites
        self.mlp = mlp
        self.wavelength_range = wavelength_range

    def scale_bijector(self, x):
        return tf.nn.softplus(x) + self.epsilon

    def distribution(self, params):
        loc, scale = tf.unstack(params)
        scale = self.scale_bijector(scale)
        q = tfd.Normal(loc, scale)
        return q

    def build(self, shapes):
        dmodel = shapes[-1]
        self.input_layer = tfk.layers.Dense(
            dmodel,
            kernel_initializer='glorot_normal',
            use_bias=False,
        )
        self.output_layer = tfk.layers.EinsumDense(
            '...d,dab->...ab',
            output_shape=(2, self.num_atoms),
            kernel_initializer='glorot_normal',
        )

    def _distribution(self, loc, scale):
        q = tfd.Normal(
            loc, 
            scale, 
        )
        return q

    def distribution(self, asu_id, hkl):
        loc = self.rac.gather(self.loc, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        q = self._distribution(loc, scale)
        return q

    def flat_distribution(self=None):
        q = self._distribution(self.loc, self.scale)
        return q

    def call(self, inputs=None):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        wav = self.mlp(wavelength.flat_values)
        out = self.mlp(wav)
        out = self.decod(wav)
