import numpy as np
import tensorflow as tf
from abismal.distributions import FoldedNormal,Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.symmetry import Op,ReciprocalASUCollection
from abismal.prior.base import PriorBase


@tfk.saving.register_keras_serializable(package="abismal")
class WilsonPriorBase(PriorBase):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, rac, sigma=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. If this is an array it must be length rac.asu_size
        """
        super().__init__(**kwargs)
        self.rac = rac
        self._sigma = sigma
        self.built = True #This is always true

    @property
    def sigma(self):
        return self._sigma

    @staticmethod
    def _empirical_binned_mean(rac, dataset, bins=20, maxiter=100, isigi_cutoff=0.0):
        """Accumulate the inverse-variance-weighted mean intensity per resolution bin.

        Returns
        -------
        mean : tf.Tensor
            Weighted mean intensity in each of `bins` resolution shells.
        size : tf.Tensor
            Summed inverse-variance weight in each shell (an effective count).
        labels : array
            Per-reflection bin assignment for `rac.dHKL`.
        edges : array
            The `bins + 1` resolution (dHKL) bin edges.
        """
        from reciprocalspaceship.utils import bin_by_percentile
        labels,edges = bin_by_percentile(rac.dHKL, bins=bins, ascending=False)
        mean = tf.zeros(bins)
        size = tf.zeros(bins)
        from tqdm import tqdm
        for i,(batch, _) in tqdm(enumerate(dataset), total=maxiter):
            if i > maxiter:
                break
            (
                asu_id,
                hkl_in,
                resolution,
                wavelength,
                metadata,
                iobs,
                sigiobs,
            ) = batch
            idx = rac.gather(labels, asu_id, hkl_in).flat_values
            I = tf.squeeze(iobs.flat_values, axis=-1)
            SIGI = tf.squeeze(sigiobs.flat_values, axis=-1)

            if isigi_cutoff is not None:
                mask = tf.cast(I / SIGI > isigi_cutoff, 'float32')
            else:
                mask = tf.ones_like(I)

            # Inverse-variance weight per observation; masked observations get zero weight.
            w = mask / tf.square(SIGI)

            # `size` now accumulates summed weights rather than counts. The batched
            # Welford combine rule mean += (W_b / W) * (mean_b - mean) holds for
            # weighted means when W is the running sum of weights.
            batch_size = tf.scatter_nd(idx[:,None], w, (bins,))
            # divide_no_nan guards bins with zero summed weight in this batch
            # (e.g. a resolution shell absent from the batch, or fully masked
            # by the I/sigI cutoff). Such a bin gets batch_mean=0 and an update
            # weight of 0, leaving the running mean untouched instead of
            # poisoning it with NaN (0/0 -> NaN, then 0 * NaN -> NaN).
            batch_mean = tf.math.divide_no_nan(
                tf.scatter_nd(idx[:,None], w * I, (bins,)), batch_size
            )

            size = size + batch_size
            mean = mean + tf.math.divide_no_nan(batch_size, size) * (batch_mean - mean)

        return mean, size, labels, edges

    @classmethod
    def with_empirical_sigma(cls, rac, dataset, bins=20, maxiter=100, isigi_cutoff=0.0, standardize=True, interpolate=True, **kwargs):
        mean, size, labels, edges = cls._empirical_binned_mean(
            rac, dataset, bins=bins, maxiter=maxiter, isigi_cutoff=isigi_cutoff
        )

        if standardize:
            k = tf.math.reduce_sum(
                mean * size / tf.math.reduce_sum(size)
            )
            mean = mean / k

        if interpolate:
            from scipy.interpolate import interp1d
            eps = 1e-3
            x = np.concatenate((
                [edges[0]+eps], 
                0.5 * (edges[1:] + edges[:-1]),
                [edges[-1]-eps],
            ))
            y = np.concatenate(([mean[0]], mean, [mean[-1]]))
            sigma = interp1d(x**-2., y)(rac.dHKL**-2)
            sigma = tf.cast(sigma, 'float32')
        else:
            sigma = tf.gather(mean, labels)

        return cls(rac, sigma)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'sigma' : tfk.saving.serialize_keras_object(self.sigma),
        })
        return config


    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        config['sigma'] = tfk.saving.deserialize_keras_object(config['sigma'])
        return cls(**config)


class AutoWilsonPriorBase(WilsonPriorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        b_init = 0.
        self.b = self.add_weight('WilsonB', shape=(), initializer=tfk.initializers.Constant(b_init))
        self._k = self.add_weight('WilsonK', shape=(), initializer='zeros')

    @property
    def k(self):
        return tf.math.exp(self._k)
        #return tf.nn.softplus(self._k)

    @property
    def sigma(self):
        sigma = tf.math.exp(-self.b * tf.math.reciprocal(tf.math.square(self.rac.dHKL)) + self._k)
        #sigma = sigma / tf.math.reduce_mean(sigma)
        return sigma

    def call(self, inputs, flat):
        self.add_metric(self.k, 'WilsonK')
        self.add_metric(self.b, 'WilsonB')
        return super().call(inputs, flat)

