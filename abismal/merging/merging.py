import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gemmi
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from abismal.symmetry import Op
import tf_keras as tfk
#from abismal.layers import Standardize,Normalize
from abismal.layers.normalization import StandardizeIntensities,StandardizeMetadata

def to_indexed_slices(tensor):
    """
    This is used to sparsify the structure factor gradients. 
    """
    mask = tensor != 0.
    shape = tfk.backend.shape(tensor)
    idx = tf.where(mask)
    idx = tf.squeeze(idx, axis=-1)
    result = tf.IndexedSlices(
        tf.boolean_mask(tensor, mask),
        idx,
        shape,
    )
    return result

@tfk.saving.register_keras_serializable(package="abismal")
class VariationalMergingModel(tfk.models.Model):
    def __init__(
            self, 
            scale_model, 
            surrogate_posterior, 
            prior, 
            likelihood, 
            mc_samples=1, 
            kl_weight=1., 
            epsilon=1e-6, 
            reindexing_ops=None, 
            standardize=False,
            **kwargs):
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
        self.standardize = standardize
        if standardize:
            self.standardize_intensity = StandardizeIntensities()
            self.standardize_metadata = StandardizeMetadata()
        else:
            self.standardize_intensity = None
            self.standardize_metadata = None

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
            'standardize' : self.standardize,
        })
        for k in ['scale_model', 'surrogate_posterior', 'likelihood', 'prior']:
            config[k] = tfk.saving.serialize_keras_object(config[k])
        # standardize_intensity/standardize_metadata are plain tracked attributes
        # (not part of __init__'s config), so they must be serialized explicitly
        # here -- otherwise Keras never calls build_from_config() on them and
        # they arrive at weight-loading time with zero variables built.
        if self.standardize_intensity is not None:
            config['standardize_intensity'] = tfk.saving.serialize_keras_object(self.standardize_intensity)
        if self.standardize_metadata is not None:
            config['standardize_metadata'] = tfk.saving.serialize_keras_object(self.standardize_metadata)
        return config

    @classmethod
    def from_config(cls, config):
        standardize_intensity = config.pop('standardize_intensity', None)
        standardize_metadata = config.pop('standardize_metadata', None)
        for k in ['scale_model', 'surrogate_posterior', 'likelihood', 'prior']:
            config[k] = tfk.saving.deserialize_keras_object(config[k])
        instance = cls(**config)
        if standardize_intensity is not None:
            instance.standardize_intensity = tfk.saving.deserialize_keras_object(standardize_intensity)
        if standardize_metadata is not None:
            instance.standardize_metadata = tfk.saving.deserialize_keras_object(standardize_metadata)
        return instance

    def build(self, shapes):
        if self.built:
            return
        self.scale_model.build(shapes)
        if self.standardize_intensity  is not None:
            self.standardize_intensity.build(shapes[-1])
        if self.standardize_metadata is not None:
            self.standardize_metadata.build(shapes[-3])
        self.built = True

    def standardize_inputs(self, inputs, training=None):
        out = inputs
        if self.standardize_intensity is not None:
            out = self.standardize_intensity(out, training=training) 
        if self.standardize_metadata is not None:
            out = self.standardize_metadata(out, training=training)
        return out

    def call(self, inputs, mc_samples=None, training=None, **kwargs):
        if self.surrogate_posterior.independent:
            return self._call_independent(inputs, mc_samples=mc_samples, training=training, **kwargs)
        return self._call_dependent(inputs, mc_samples=mc_samples, training=training, **kwargs)

    def _call_dependent(self, inputs, mc_samples=None, training=None, **kwargs):
        """
        Used for posteriors in which different miller indices are statistically dependent such as the MultivariateNormalPosterior
        """
        if mc_samples is None:
            mc_samples = self.mc_samples

        inputs = self.standardize_inputs(inputs, training=training)

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
        hkl = None
        kl_div = None
        q = self.surrogate_posterior.flat_distribution()
        z = q.sample(mc_samples)

        p = self.prior(inputs, flat=True)
        kl_div = self.surrogate_posterior.compute_kl_terms(q, p, samples=z)

        for op in self.reindexing_ops:
            _hkl = tf.ragged.map_flat_values(op, hkl_in)
            _z = self.surrogate_posterior.rac.gather(
                tf.transpose(z),
                asu_id.flat_values,
                _hkl.flat_values,
            )

            _ipred = tf.RaggedTensor.from_row_splits(_z, iobs.row_splits)

            if self.surrogate_posterior.parameterization == 'structure_factor':
                _ipred = tf.square(_ipred)
            _ipred = _ipred * scale

            _ll = tf.ragged.map_flat_values(self.likelihood, _ipred, iobs, sigiobs)
            _ll = tf.reduce_mean(_ll, [-1, -2], keepdims=True)

            if ll is None:
                ipred = _ipred
                ll = _ll
                hkl = _hkl
            else:
                idx =  _ll > ll
                ipred = tf.where(idx, _ipred, ipred)
                ll = tf.where(idx, _ll, ll)
                hkl = tf.where(idx, _hkl, hkl)

        if training:
            self.surrogate_posterior.register_seen(asu_id.flat_values, hkl.flat_values)

        self.likelihood.register_metrics(
            ipred.flat_values, 
            iobs.flat_values, 
            sigiobs.flat_values,
        )

        ll = tf.reduce_mean(ll)
        kl_div = tf.reduce_mean(kl_div) 

        self.add_metric(-ll, name='NLL')
        self.add_loss(-ll)

        self.add_metric(kl_div, name='KL')
        self.add_loss(self.kl_weight * kl_div)

        ipred_avg = tf.reduce_mean(ipred, axis=-1)
        return ipred_avg

    def _call_independent(self, inputs, mc_samples=None, training=None, **kwargs):
        """
        The default method can be used for most posteriors
        """
        if mc_samples is None:
            mc_samples = self.mc_samples

        inputs = self.standardize_inputs(inputs, training=training)

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
        hkl = None
        kl_div = None

        for op in self.reindexing_ops:
            _hkl = tf.ragged.map_flat_values(op, hkl_in)
            q = self.surrogate_posterior.distribution(asu_id.flat_values, _hkl.flat_values)
            _inputs = inputs[:1] + (_hkl,) + inputs[2:]
            p = self.prior(_inputs, flat=False)
            z = q.sample(mc_samples)
            _kl_div = self.surrogate_posterior.compute_kl_terms(q, p, samples=z)
 
            _kl_div = tf.RaggedTensor.from_row_splits(_kl_div[...,None], iobs.row_splits)
            _ipred = tf.RaggedTensor.from_row_splits(tf.transpose(z), iobs.row_splits)

            if self.surrogate_posterior.parameterization == 'structure_factor':
                _ipred = tf.square(_ipred)
            _ipred = _ipred * scale

            _ll = tf.ragged.map_flat_values(self.likelihood, _ipred, iobs, sigiobs)
            _ll = tf.reduce_mean(_ll, [-1, -2], keepdims=True)

            if ll is None:
                ipred = _ipred
                ll = _ll
                hkl = _hkl
                kl_div = _kl_div
            else:
                idx =  _ll > ll
                ipred = tf.where(idx, _ipred, ipred)
                ll = tf.where(idx, _ll, ll)
                hkl = tf.where(idx, _hkl, hkl)
                kl_div = tf.where(idx, _kl_div, kl_div)

        if training:
            self.surrogate_posterior.register_seen(asu_id.flat_values, hkl.flat_values)

        self.likelihood.register_metrics(
            ipred.flat_values, 
            iobs.flat_values, 
            sigiobs.flat_values,
        )

        ll = tf.reduce_mean(ll)

        kl_bins = 5
        #kl_quants = tfp.stats.quantiles(kl_div.flat_values, kl_bins)
        #for i in range(kl_bins):
        #    self.add_metric(kl_quants[i], f'KL_{i+1}')


        self.add_metric(-ll, name='NLL')
        self.add_loss(-ll)

        #self.add_metric(tf.reduce_mean(kl_div), name='KL')                       # true mean, unfloored
        #self.add_metric(tfp.stats.percentile(kl_div, 50.), name='KL_med')        # for anchoring λ
        #free_bits = 5.0                                                          # nats/reflection
        #self.add_loss(self.kl_weight * tf.reduce_mean(tf.maximum(kl_div, free_bits)))
        #self.add_metric(tf.reduce_mean(tf.cast(kl_div <= free_bits, 'float32')), 'NumSaturated')

        kl_div = tf.reduce_mean(kl_div) 
        self.add_metric(kl_div, name='KL')
        self.add_loss(self.kl_weight * kl_div)

        ipred_avg = tf.reduce_mean(ipred, axis=-1)
        return ipred_avg

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
        self.optimizer.lazy_vars = [v._unique_id for v in q_vars]

        grad_q = tape.gradient(loss, q_vars)
        grad_q_norm = tf.sqrt(
            tf.reduce_mean([tf.reduce_mean(tf.square(g)) for g in grad_q])
        )
        metrics["|∇q|"] = grad_q_norm
        #grad_q = [to_indexed_slices(g) for g in grad_q] #This makes lazy adam work

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

        p_vars = self.prior.trainable_variables
        if len(p_vars) > 0:
            grad_p = tape.gradient(loss, p_vars)
            grad_p_norm = tf.sqrt(
                tf.reduce_mean([tf.reduce_mean(tf.square(g)) for g in grad_p])
            )
            trainable_vars += p_vars
            gradients += grad_p
            metrics["|∇p|"] = grad_p_norm


        gradients = [tf.where(tf.math.is_finite(g), g, 0.) for g in gradients]
        # Apply in trainable_variables order so the optimizer creates its slot
        # variables in the same order load_model rebuilds them (it builds the
        # optimizer against model.trainable_variables). The per-group order above
        # (scale, q, ll, p) otherwise differs from tracking order whenever the
        # prior is trainable (AutoWilsonPrior), and the saved optimizer slots then
        # deserialize into the wrong variables -- a shape mismatch on load.
        grad_by_var = {v.ref(): g for v, g in zip(trainable_vars, gradients)}
        ordered = [
            (grad_by_var[v.ref()], v)
            for v in self.trainable_variables if v.ref() in grad_by_var
        ]
        self.optimizer.apply_gradients(ordered)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return metrics


