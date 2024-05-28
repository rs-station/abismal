import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
import tensorflow_probability as tfp
import gemmi
from abismal.layers.standardization import Standardize
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from abismal.symmetry import Op
import tf_keras as tfk
from IPython import embed


class VariationalMergingModel(tfk.models.Model):
    def __init__(self, scale_model, surrogate_posterior, likelihood, mc_samples=1, eps=1e-6, reindexing_ops=None):
        super().__init__()
        self.eps = eps
        self.likelihood = likelihood
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.mc_samples = mc_samples
        if reindexing_ops is None:
            reindexing_ops = ["x,y,z"]
        self.reindexing_ops = [Op(op) for op in reindexing_ops]
        self.standardize_intensity = Standardize(center=False)
        self.standardize_metadata = Standardize()

    def call(self, inputs, mc_samples=None, **kwargs):
        if mc_samples is None:
            mc_samples = self.mc_samples

        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        if self.standardize_intensity is not None:
            iobs = tf.ragged.map_flat_values(self.standardize_intensity, iobs)
            sigiobs = tf.ragged.map_flat_values(self.standardize_intensity.standardize, sigiobs)
        if self.standardize_metadata is not None:
            metadata = tf.ragged.map_flat_values(self.standardize_metadata, metadata)

        training = kwargs.get('training', None)
        ll = None
        ipred = None

        q = self.surrogate_posterior.flat_distribution()
        ipred = q.sample(mc_samples)

        kl_div = self.surrogate_posterior.register_kl(ipred, asu_id, hkl, training)

        if self.surrogate_posterior.parameterization == 'structure_factor':
            ipred = tf.square(ipred)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)

        ipred = tf.transpose(ipred)
        ipred_scaled = None

        _inputs = (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) 
        scale = self.scale_model(
            _inputs, 
            mc_samples=mc_samples, 
            **kwargs
        )

        for op in self.reindexing_ops:
            # Choose the best indexing solution for each image
            _hkl = tf.ragged.map_flat_values(op, hkl)

            _ipred = self.surrogate_posterior.rac.gather(ipred, asu_id, _hkl)
            _ipred = _ipred * scale

            _ll = tf.ragged.map_flat_values(self.likelihood, _ipred, iobs, sigiobs)
            _ll = tf.reduce_mean(_ll, [-1, -2], keepdims=True)

            if ll is None:
                ipred_scaled = _ipred
                ll = _ll
            else:
                idx =  _ll > ll
                ipred_scaled = tf.where(idx, _ipred, ipred_scaled)
                ll = tf.where(idx, _ll, ll)

        self.likelihood.register_metrics(
            ipred_scaled.flat_values, 
            iobs.flat_values, 
            sigiobs.flat_values,
        )

        # This is the mean across mc samples and observations
        ll = tf.reduce_mean(ll) 

        self.add_metric(-ll, name='NLL')
        self.add_loss(-ll)

        ipred_avg = tf.reduce_mean(ipred_scaled, axis=-1)
        return ipred_avg

    #For production with super nan avoiding powers
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # Set up metrics dict
        metrics = {m.name: m.result() for m in self.metrics}

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        scale_vars = self.scale_model.trainable_variables
        grad_scale = tape.gradient(loss, scale_vars)
        grad_s_norm = tf.sqrt(
            tf.reduce_mean([tf.reduce_mean(tf.square(g)) for g in grad_scale])
        )
        metrics["|∇s|"] = grad_s_norm

        q_vars = self.surrogate_posterior.trainable_variables
        grad_q= tape.gradient(loss, q_vars)
        grad_q_norm = tf.sqrt(
            tf.reduce_mean([tf.reduce_mean(tf.square(g)) for g in grad_q])
        )
        metrics["|∇q|"] = grad_q_norm

        trainable_vars = scale_vars + q_vars 

        gradients = grad_scale + grad_q 

        ll_vars = self.likelihood.trainable_variables
        if len(ll_vars) > 0:
            grad_ll = tape.gradient(loss, ll_vars)
            grad_ll_norm = tf.sqrt(
                tf.reduce_mean([tf.reduce_mean(tf.square(g)) for g in grad_ll])
            )
            trainable_vars += ll_vars
            gradients += grad_ll
            metrics["|∇ll|"] = grad_ll_norm

        gradients = [tf.where(tf.math.is_finite(g), g, 0.) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)


        # Return a dict mapping metric names to current value
        return metrics


