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
        self.sigma = sigma
        self.built = True #This is always true

    @classmethod
    def with_empirical_sigma(cls, rac, dataset, bins=20, maxiter=100, isigi_cutoff=0.0, standardize=True, **kwargs):
        from reciprocalspaceship.utils import bin_by_percentile
        labels,edges = bin_by_percentile(rac.dHKL, ascending=False)
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

            if isigi_cutoff is not None:
                SIGI = tf.squeeze(sigiobs.flat_values, axis=-1)
                mask = tf.cast(I / SIGI > isigi_cutoff, 'float32')
            else:
                mask = tf.one_like(I)

            batch_size = tf.scatter_nd(idx[:,None], mask, (20,))
            batch_mean = tf.scatter_nd(idx[:,None], I, (20,)) / batch_size

            size = size + batch_size
            mean = mean + (batch_size / size) * (batch_mean - mean)

        if standardize:
            mean = mean / tf.math.reduce_std(mean)
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


