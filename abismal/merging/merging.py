import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from IPython import embed
from abismal_zero.layers import *
from abismal_zero.blocks import *


def spearman_cc(yobs, ypred):
    from scipy.stats import spearmanr
    from scipy.special import seterr
    seterr(all='ignore')

    cc = spearmanr(yobs, ypred)[0]

    if not np.isfinite(cc):
        cc = 0.
    return cc


class VariationalMergingModel(tfk.models.Model):
    def __init__(self, scale_model, surrogate_posterior, studentt_dof=None, sigiobs_model=None, mc_samples=1, eps=1e-6, use_global_scale=True):
        super().__init__()
        self.eps = eps
        self.dof = studentt_dof
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.sigiobs_model = sigiobs_model
        self.mc_samples = mc_samples
        if use_global_scale:
            self.global_scale = tfu.TransformedVariable(
                1., 
                tfb.Chain([
                    tfb.Shift(eps), 
                    tfb.Softplus()
                ]),
            )
        else:
            self.global_scale = 1.

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

        ipred = self.surrogate_posterior(hkl, mc_samples=mc_samples)

        iscale = self.surrogate_posterior.stddev(hkl)
        iloc   = self.surrogate_posterior.mean(hkl)
        imodel = tf.concat((
            iloc[...,None],
            iscale[...,None],
        ),  axis=-1)

        scale = self.scale_model((
            metadata, 
            iobs,
            sigiobs,
            imodel,
        ), mc_samples=mc_samples, **kwargs)

        ipred = self.global_scale * ipred * scale

        if self.sigiobs_model is not None:
            sigiobs_pred = self.sigiobs_model(sigiobs, ipred)
        else:
            sigiobs_pred = sigiobs

        R = iobs - ipred

        if self.dof is None:
            ll = tfd.Normal(0., sigiobs_pred.flat_values).log_prob(R.flat_values)
        else:
            ll = tfd.StudentT(self.dof, 0, sigiobs_pred.flat_values).log_prob(R.flat_values)

        # Clampy clamp clamp
        ll = tf.maximum(ll, -1e4)

        # This is the mean factoring the mask and any mc samples
        ll = tf.reduce_mean(ll) 

        self.add_metric(-ll, name='NLL')
        self.add_loss(-ll)

        # This is the mean ipred across the posterior mc samples
        ipred = tf.reduce_mean(ipred, axis=-1)
        cc = tf.numpy_function(spearman_cc, [iobs.flat_values, ipred.flat_values], Tout=tf.float64)
        self.add_metric(cc, name='CCpred')

        return ipred

    #For production with super nan avoiding powers
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [tf.where(tf.math.is_finite(g), g, 0.) for g in gradients]

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        # Record the norm of the gradients
        grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in gradients]))
        metrics['grad_norm'] = grad_norm
        return metrics


