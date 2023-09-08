import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
import tensorflow_probability as tfp
import gemmi
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


class Op(tfk.layers.Layer):
    def __init__(self, triplet):
        super().__init__()
        self.gemmi_op = gemmi.Op(triplet)
        self.rot = tf.convert_to_tensor(self.gemmi_op.rot, dtype='float32')
        self.den = tf.convert_to_tensor(self.gemmi_op.DEN, dtype='float32')
        self.identity = self.gemmi_op == 'x,y,z'

    def __str__(self):
        return f"Op({self.gemmi_op.triplet()})"

    def call(self, hkl):
        if self.identity:
            return hkl
        dtype = hkl.dtype
        hkl = tf.cast(hkl, tf.float32)
        hkl = tf.math.floordiv(tf.matmul(hkl, self.rot), self.den)
        hkl = tf.cast(hkl, dtype)
        return hkl

class VariationalMergingModel(tfk.models.Model):
    def __init__(self, scale_model, surrogate_posterior, studentt_dof=None, sigiobs_model=None, mc_samples=1, eps=1e-6, reindexing_ops=None):
        super().__init__()
        self.eps = eps
        self.dof = studentt_dof
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.sigiobs_model = sigiobs_model
        self.mc_samples = mc_samples
        if reindexing_ops is None:
            reindexing_ops = ["x,y,z"]
        self.reindexing_ops = [Op(op) for op in reindexing_ops]

    def _likelihood(self):
        pass

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

        training = kwargs.get('training', None)
        ll = None
        ipred = None

        q = self.surrogate_posterior.flat_distribution()
        ipred = q.sample(mc_samples)
        kl_div = self.surrogate_posterior.register_kl(ipred, asu_id, hkl, training)
        #if not tf.math.is_finite(kl_div):
        #    from IPython import embed
        #    embed(colors='linux')

        if self.surrogate_posterior.parameterization == 'structure_factor':
            ipred = tf.square(ipred)
            floc = q.mean()[...,None]
            fscale = q.stddev()[...,None]
            # Exact
            iloc = floc * floc + fscale * fscale
            # 1st order
            iscale = 2 * floc * fscale
            imodel = (
                floc,
                fscale,
                iloc,
                iscale,
            )
        else:
            iloc = q.mean()[...,None]
            iscale = q.stddev()[...,None]
            imodel = (
                iloc,
                iscale,
            )
        imodel = tf.concat(imodel, axis=-1)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)

        ipred = tf.transpose(ipred)

        for op in self.reindexing_ops:
            # Choose the best indexing solution for each image
            _hkl = tf.ragged.map_flat_values(op, hkl)

            _ipred = self.surrogate_posterior.rac.gather(ipred, asu_id, _hkl)
            _imodel = self.surrogate_posterior.rac.gather(imodel, asu_id, _hkl)

            _inputs = (
                asu_id,
                _hkl,
                resolution,
                wavelength,
                metadata,
                iobs,
                sigiobs,
            ) 

            scale = self.scale_model(
                _inputs, 
                _imodel,
                mc_samples=mc_samples, 
                **kwargs
            )

            _ipred = _ipred * scale

            if self.sigiobs_model is not None:
                sigiobs_pred = self.sigiobs_model(sigiobs, _ipred)
            else:
                sigiobs_pred = sigiobs

            R = iobs - _ipred

            if self.dof is None:
                _ll = tfd.Normal(0., sigiobs_pred.flat_values).log_prob(R.flat_values)
            else:
                _ll = tfd.StudentT(self.dof, 0, sigiobs_pred.flat_values).log_prob(R.flat_values)

            _ll = tf.RaggedTensor.from_value_rowids(
                _ll, 
                _ipred.value_rowids(),
            )
            _ll = tf.reduce_mean(_ll, [-1, -2], keepdims=True)

            if ll is None:
                ipred = _ipred
                ll = _ll
            else:
                idx =  _ll > ll
                ipred = tf.where(idx, _ipred, ipred)
                ll = tf.where(idx, _ll, ll)


        # This is the mean across mc samples and observations
        ll = tf.reduce_mean(ll) 
        #if not tf.math.is_finite(ll):
        #    from IPython import embed
        #    embed(colors='linux')

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


