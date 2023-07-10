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

def weighted_pearsonr(x, y, w):
    """
    Calculate a [weighted Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient).

    Note
    ----
    x, y, and w may have arbitrarily shaped leading dimensions. The correlation coefficient will always be computed pairwise along the last axis.

    Parameters
    ----------
    x : np.array(float)
        An array of observations.
    y : np.array(float)
        An array of observations the same shape as x.
    w : np.array(float)
        An array of weights the same shape as x. These needn't be normalized.

    Returns
    -------
    r : float
        The Pearson correlation coefficient along the last dimension. This has shape {x,y,w}.shape[:-1].
    """
    z = tf.math.reciprocal(
        tf.reduce_sum(w, axis=-1, keepdims=True)
    )

    mx = z * tf.reduce_sum(w * x, axis=-1, keepdims=True)
    my = z * tf.reduce_sum(w * y, axis=-1, keepdims=True)

    dx = x - mx
    dy = y - my

    cxy = z * tf.reduce_sum(w * dx * dy, axis=-1)
    cx  = z * tf.reduce_sum(w * dx * dx, axis=-1)
    cy  = z * tf.reduce_sum(w * dy * dy, axis=-1)

    r = cxy / tf.sqrt(cx * cy)
    return r

def spearman_cc(yobs, ypred):
    from scipy.stats import spearmanr
    from scipy.special import seterr
    seterr(all='ignore')

    cc = spearmanr(yobs, ypred)[0]

    if not np.isfinite(cc):
        cc = 0.
    return cc


class VariationalMergingModel(tfk.models.Model):
    def __init__(self, scale_model, surrogate_posterior, studentt_dof=None, 
            sigiobs_model=None, mc_samples=1, eps=1e-6, clamp=None, rescale=True,
            ):
        super().__init__()
        self.eps = eps
        self.dof = studentt_dof
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.sigiobs_model = sigiobs_model
        self.mc_samples = mc_samples
        self.clamp = clamp
        self.rescale = rescale

    def call(self, inputs, mc_samples=None, **kwargs):
        if mc_samples is None: mc_samples = self.mc_samples

        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        rescale_factor = 1.
        if self.rescale:
            rescale_factor = tf.math.reduce_std(iobs, axis=-2, keepdims=True)

        ipred = self.surrogate_posterior(asu_id.flat_values, hkl.flat_values, mc_samples=self.mc_samples)

        if self.surrogate_posterior.parameterization == 'structure_factor':
            aid = tf.squeeze(asu_id, axis=-1)
            floc   = tf.ragged.map_flat_values(self.surrogate_posterior.mean, aid, hkl)
            fscale = tf.ragged.map_flat_values(self.surrogate_posterior.stddev, aid, hkl)
            iloc  = tf.square(floc) + tf.square(fscale)
            iscale= tf.abs(2. * floc * fscale)
            imodel = (
                rescale_factor * iloc[...,None],
                rescale_factor * iscale[...,None],
                #rescale_factor * floc[...,None],
                #rescale_factor * fscale[...,None],
            )
        elif self.surrogate_posterior.parameterization == 'intensity':
            aid = tf.squeeze(asu_id, axis=-1)
            #iloc   = self.surrogate_posterior.mean(aid, hkl)
            #iscale = self.surrogate_posterior.stddev(aid, hkl)
            iloc   = tf.ragged.map_flat_values(self.surrogate_posterior.mean, aid, hkl)
            iscale = tf.ragged.map_flat_values(self.surrogate_posterior.stddev, aid, hkl)
            imodel = (
                rescale_factor * iloc[...,None],
                rescale_factor * iscale[...,None],
            )
        else:
            raise AttributeError(
                "Surrogate posteriors must have attribute 'parameterization' with value "
                "of 'intensity' or 'structure_factor'                                   "
            )
        ipred = tf.RaggedTensor.from_row_splits(
            tf.transpose(ipred),
            iobs.row_splits,
        )

        imodel = tf.concat(imodel, axis=-1)
        scale = self.scale_model((
            metadata, 
            rescale_factor * iobs,
            rescale_factor * sigiobs,
            imodel,
        ), mc_samples=mc_samples, **kwargs) / rescale_factor

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
        #cc = tf.numpy_function(spearman_cc, [iobs.flat_values, ipred.flat_values], Tout=tf.float64)
        x = tf.squeeze(iobs.flat_values, axis=-1)
        y = ipred.flat_values
        w = tf.squeeze(tf.math.reciprocal(tf.square(sigiobs.flat_values)), axis=-1)
        cc = weighted_pearsonr(x, y, w)
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


