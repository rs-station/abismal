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


def spearman_cc(yobs, ypred):
    from scipy.stats import spearmanr
    from scipy.special import seterr
    seterr(all='ignore')

    cc = spearmanr(yobs, ypred)[0]

    if not np.isfinite(cc):
        cc = 0.
    return cc


class VariationalMergingModel(tfk.models.Model):
    def __init__(self, scale_model, surrogate_posterior, studentt_dof=None, sigiobs_model=None, mc_samples=1, eps=1e-6, clamp=None):
        super().__init__()
        self.eps = eps
        self.dof = studentt_dof
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.sigiobs_model = sigiobs_model
        self.mc_samples = mc_samples
        self.clamp = clamp

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


        q = self.surrogate_posterior(asu_id.flat_values, hkl.flat_values)
        if self.surrogate_posterior.parameterization == 'structure_factor':
            floc   = self.surrogate_posterior.mean(asu_id, hkl)
            fscale = self.surrogate_posterior.stddev(asu_id, hkl)
            iloc  = tf.square(floc) + tf.square(fscale)
            iscale= tf.abs(2. * floc * fscale)
            imodel = (
                iloc[...,None],
                iscale[...,None],
                floc[...,None],
                fscale[...,None],
            )
            ipred = q.sample_intensities(self.mc_samples)
        elif self.surrogate_posterior.parameterization == 'intensity':
            iloc   = self.surrogate_posterior.mean(asu_id, hkl)
            iscale = self.surrogate_posterior.stddev(asu_id, hkl)
            imodel = (
                iloc[...,None],
                iscale[...,None],
            )
            ipred = q.sample(self.mc_samples)
        else:
            raise AttributeError(
                "Surrogate posteriors must have attribute 'parameterization' with value "
                "of 'intensity' or 'structure_factor'                                   "
            )
        ipred = tf.RaggedTensor.from_row_splits(
            tf.transpose(ipred),
            iobs.row_splits,
        )

        iscale = self.surrogate_posterior.stddev(asu_id, hkl)
        iloc   = self.surrogate_posterior.mean(asu_id, hkl)
        imodel = tf.concat(imodel, axis=-1)

        scale = self.scale_model((
            metadata, 
            iobs,
            sigiobs,
            imodel,
        ), mc_samples=mc_samples, **kwargs)

        ipred = ipred * scale

        if self.sigiobs_model is not None:
            sigiobs_pred = self.sigiobs_model(sigiobs, ipred)
        else:
            sigiobs_pred = sigiobs

        R = iobs - ipred

        if self.dof is None:
            ll = tfd.Normal(0., sigiobs_pred.flat_values).log_prob(R.flat_values)
        else:
            ll = tfd.StudentT(self.dof, 0, sigiobs_pred.flat_values).log_prob(R.flat_values)

        # Clamp
        if self.clamp is not None:
            ll = tf.maximum(ll, -1e4)

        # This is the mean across mc samples and observations
        ll = tf.reduce_mean(ll) 

        self.add_metric(-ll, name='NLL')
        self.add_loss(-ll)

        # This is the mean ipred across the posterior mc samples
        ipred = tf.reduce_mean(ipred, axis=-1)
        cc = tf.numpy_function(spearman_cc, [iobs.flat_values, ipred.flat_values], Tout=tf.float64)
        self.add_metric(cc, name='CCpred')

        return ipred

    #For production with super nan avoiding powers
    def traXXin_step(self, data):
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


