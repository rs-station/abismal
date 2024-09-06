import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
import tensorflow_probability as tfp
import gemmi
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from abismal.symmetry import Op
import tf_keras as tfk
from abismal.layers import Standardize


@tfk.saving.register_keras_serializable(package="abismal")
class VariationalMergingModel(tfk.models.Model):
    def __init__(self, scale_model, surrogate_posterior, prior, likelihood, mc_samples=1, kl_weight=1., epsilon=1e-6, reindexing_ops=None, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.likelihood = likelihood
        self.prior = prior
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.mc_samples = mc_samples
        self.kl_weight = kl_weight
        if reindexing_ops is None:
            reindexing_ops = ["x,y,z"]
        self.reindexing_ops = [Op(op) for op in reindexing_ops]

    def get_config(self):
        ops = self.reindexing_ops
        if ops is not None:
            ops = [op.gemmi_op.triplet() for op in self.reindexing_ops]
        config = super().get_config()

        config.update({
            'scale_model' : self.scale_model,
            'surrogate_posterior' : self.surrogate_posterior,
            'prior' : self.prior,
            'likelihood' : self.likelihood, 
            'mc_samples' : self.mc_samples,
            'kl_weight' : 1.,
            'epsilon' : self.epsilon,
            'reindexing_ops' : ops,
        })
        for k in ['scale_model', 'surrogate_posterior', 'likelihood', 'prior']:
            config[k] = tfk.saving.serialize_keras_object(config[k])
        return config

    @classmethod
    def from_config(cls, config):
        for k in ['scale_model', 'surrogate_posterior', 'likelihood']:
            config[k] = tfk.saving.deserialize_keras_object(config[k])
        return cls(**config)

    def build(self, shapes):
        self.scale_model.build(shapes)

    def call(self, inputs, mc_samples=None, training=None, **kwargs):
        if mc_samples is None:
            mc_samples = self.mc_samples

        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        scale = self.scale_model(
            inputs,
            mc_samples=mc_samples, 
            **kwargs
        )

        ll = None
        ipred = None
        kl_div = None
        hkl = None

        for op in self.reindexing_ops:
            # Choose the best indexing solution for each image
            _hkl = tf.ragged.map_flat_values(op, hkl_in)

            _q = self.surrogate_posterior.distribution(asu_id.flat_values, _hkl.flat_values)
            _p = self.prior.distribution(asu_id.flat_values, _hkl.flat_values)

            #We need these shenanigans to support multivariate posteriors
            _z = _q.sample(mc_samples)
            if _q.event_shape != []:
                _ipred = _z[...,0]
            else:
                _ipred = _z

            _kl_div = self.surrogate_posterior.compute_kl_terms(_q, _p, samples=_z)
            _kl_div = tf.RaggedTensor.from_row_splits(_kl_div[...,None], scale.row_splits)

            if self.surrogate_posterior.parameterization == 'structure_factor':
                _ipred = tf.square(_ipred)

            _ipred = tf.transpose(_ipred)
            _ipred = tf.RaggedTensor.from_row_splits(_ipred, scale.row_splits)
            _ipred = _ipred * scale

            _ll = tf.ragged.map_flat_values(self.likelihood, _ipred, iobs, sigiobs)
            _ll = tf.reduce_mean(_ll, [-1, -2], keepdims=True)

            if ll is None:
                ipred = _ipred
                ll = _ll
                kl_div = _kl_div
                hkl = _hkl
            else:
                idx =  _ll > ll
                ipred = tf.where(idx, _ipred, ipred)
                ll = tf.where(idx, _ll, ll)
                kl_div = tf.where(idx, _kl_div, kl_div)
                hkl = tf.where(idx, _hkl, hkl)

        if training:
            self.surrogate_posterior.register_seen(asu_id.flat_values, hkl.flat_values)

        self.likelihood.register_metrics(
            ipred.flat_values, 
            iobs.flat_values, 
            sigiobs.flat_values,
        )

        # This is the mean across mc samples and observations
        ll = tf.reduce_mean(ll) 
        kl_div = tf.reduce_mean(kl_div) 

        self.add_metric(-ll, name='NLL')
        self.add_loss(-ll)

        self.add_metric(kl_div, name='KL')
        self.add_loss(self.kl_weight * kl_div)

        ipred_avg = tf.reduce_mean(ipred, axis=-1)
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


