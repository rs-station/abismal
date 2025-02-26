import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from abismal.prior.base import PriorBase
import tf_keras as tfk
from tqdm import tqdm

class EmpiricalPrior(PriorBase):
    def __init__(self, rac, loc=None, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.rac =  rac
        self.loc = loc
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'rac' : tfk.saving.serialize_keras_object(self.rac),
            'loc' : tfk.saving.serialize_keras_object(self.loc),
            'scale' : tfk.saving.serialize_keras_object(self.scale),
        })
        return config

    def train(self, data, scale_model=None, num_steps=1_000):
        """
        Use Welford's algorithm to estimate the mean and
        variance of the intensities in data. Optionally you
        can apply a pre-trained scale_model. 
        """
        count = 0
        mean = tf.zeros(self.rac.asu_size)
        var_of_mean  = tf.zeros(self.rac.asu_size)
        uncertainty = tf.zeros(self.rac.asu_size)
        w_sum = tf.zeros(self.rac.asu_size)
        for count, (inputs, _) in tqdm(enumerate(data), total=num_steps):
            (
                asu_id,
                hkl_in,
                resolution,
                wavelength,
                metadata,
                iobs,
                sigiobs,
            ) = (i.flat_values for i in inputs)
                
            #TODO: does this even work??
            if scale_model is not None:
                z = tf.squeeze(scale_model(inputs, 1), axis=0)
                iobs = z * iobs
                sigiobs = z * sigiobs

            miller_id = self.rac._miller_ids(asu_id, hkl_in)
            idx = np.where(miller_id == 3000)

            iobs = tf.squeeze(iobs, axis=-1)
            sigiobs = tf.squeeze(sigiobs, axis=-1)

            sigiobs2 = tf.square(sigiobs)
            _w = tf.math.reciprocal(sigiobs2)
            _mean = self.rac.gather(mean, asu_id, hkl_in)
            _uncertainty = self.rac.gather(uncertainty, asu_id, hkl_in)

            # Update the denominator
            w_sum = tf.tensor_scatter_nd_add(w_sum, miller_id[...,None], _w)
            _w_sum = self.rac.gather(w_sum, asu_id, hkl_in)

            # Update the mean
            delta = iobs - _mean
            mean = tf.tensor_scatter_nd_add(
                tensor=mean,
                indices=miller_id[...,None],
                updates=_w * delta / _w_sum,
            )
            _mean = self.rac.gather(mean, asu_id, hkl_in)


            # Update the variance of the mean
            var_of_mean = tf.tensor_scatter_nd_add(
                tensor=var_of_mean,
                indices=miller_id[...,None],
                updates=_w * delta * (iobs - _mean),
            )

            # Update the uncertainty / variance
            delta = sigiobs2 - _uncertainty
            uncertainty = tf.tensor_scatter_nd_add(
                tensor=uncertainty,
                indices=miller_id[...,None],
                updates=_w * delta / _w_sum,
            )
            
            if count > num_steps:
                break

        # Finalize variance of mean
        var_of_mean = var_of_mean / tf.where(var_of_mean == 0., 1., w_sum)
        stddev_of_mean = tf.sqrt(var_of_mean)
        stddev_of_mean = tf.where(
            stddev_of_mean <= 0.,
            tf.reduce_max(stddev_of_mean),
            stddev_of_mean,
        )

        # Finalize uncertainty
        uncertainty = tf.sqrt(uncertainty)

        normalizer = tf.math.reduce_std(mean),
        loc = tf.where(
            uncertainty == 0., 
            tf.reduce_mean(mean),
            mean,
        )
        scale = tf.where(
            uncertainty == 0., 
            normalizer,
            uncertainty,
        )
        self.loc = loc / normalizer
        self.scale = scale / normalizer

    def distribution(self, asu_id, hkl):
        loc = self.rac.gather(self.loc, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        p = tfd.Normal(loc, scale)
        return p

    def flat_distribution(self):
        p = tfd.Normal(self.loc, self.scale)
        return p


