import numpy as np
import tensorflow as tf
import tf_keras as tfk


class RunningMoments:
    def __init__(self, axis=-2):
        self.n=0
        self.s=0
        self.mean=0
        self.axis=axis

    def update(self, x, weights=None):
        if weights is None:
            weights = np.ones_like(x)
        self.n += np.sum(weights, axis=self.axis, keepdims=True)
        diff = x - self.mean
        self.mean += np.sum(weights * diff / self.n, axis=self.axis, keepdims=True)
        self.s = self.s + np.sum(weights * diff * (x - self.mean), axis=self.axis, keepdims=True)

    @property
    def var(self):
        if np.sum(self.n) <= 1:
            return None
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt(self.var)

@tfk.saving.register_keras_serializable(package="abismal")
class Standardize(tfk.layers.Layer):
    def __init__(self, center=True, max_counts=np.inf, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts

    def get_config(self):
        conf = super().get_config()
        conf.update({
            'center' : self.center,
            'max_counts' : self.max_counts,
            'epsilon' : self.epsilon,
        })
        return conf

    def build(self, shape):
        if self.built:
            return
        mom_shape = [1] * len(shape)
        mom_shape[-1] = shape[-1]
        reduce_dims = list(range(len(shape)-1))
        self.reduce_dims = reduce_dims
        self.axis_size = shape[-1]

        self.mean = self.add_weight(
            shape=mom_shape,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='mean',
        )
        self.m2= self.add_weight(
            shape=mom_shape,
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='m2',
        )
        self.count = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.float32,
            trainable=False,
            name='count',
        )
        self.frozen = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.bool,
            trainable=False,
            name='frozen',
        )
        super().build(shape)

    @staticmethod
    def _flat(x):
        """
        Return the dense values backing `x`. Ragged tensors are unpacked to their
        flat_values so that the moment math never broadcasts a ragged operand
        against a dense one -- that lowers to `RaggedRange`, which has no XLA
        kernel and breaks `jit_compile=True`.
        """
        if isinstance(x, tf.RaggedTensor):
            return x.flat_values
        return x

    @staticmethod
    def _reduce_leading(x):
        """Sum over every axis but the last."""
        return tf.reduce_sum(x, axis=list(range(len(x.shape) - 1)))

    @property
    def flat_mean(self):
        """`mean` shaped (d,) so it broadcasts against flat values of any rank."""
        return tf.reshape(self.mean, (-1,))

    @property
    def flat_std(self):
        return tf.reshape(self.std, (-1,))

    @property
    def count_float(self):
        return tf.cast(self.count, self.mean.dtype)

    @property
    def std(self):
        return tf.sqrt(self.var)

    @property
    def var(self):
        m2 = tf.clip_by_value(self.m2, self.epsilon, np.inf)
        return m2 / self.count_float

    def update(self, x):
        x = self._flat(x)
        k = tf.reduce_sum(tf.ones_like(x)) / self.axis_size
        self.count.assign_add(k)
        mean = self.flat_mean
        diff = x - mean
        new_mean = mean + self._reduce_leading(diff) / self.count
        diff *= (x - new_mean)
        self.mean.assign(tf.reshape(new_mean, self.mean.shape))
        self.m2.assign(
            self.m2 + tf.reshape(self._reduce_leading(diff), self.m2.shape)
        )

    def standardize(self, data):
        if isinstance(data, tf.RaggedTensor):
            return data.with_flat_values(self.standardize(data.flat_values))
        if self.center:
            return (data - self.flat_mean) / self.flat_std
        return data / self.flat_std

    def _update_if_unfrozen(self, x):
        self.update(x)
        return tf.constant(0)

    def call(self, data, training=True):
        if self.max_counts > self.max_counts:
             training = False
        if training:
            # `self.frozen` is a variable (not a Python bool) so this branch is
            # read at graph *execution* time. That lets StandardizationFreezer
            # toggle it mid-training without forcing a retrace of the compiled
            # train_step -- a plain Python `if self.frozen` here would get
            # baked into the graph at first trace and never update again.
            tf.cond(self.frozen, lambda: tf.constant(0), lambda: self._update_if_unfrozen(data))
        return self.standardize(data)

@tfk.saving.register_keras_serializable(package="abismal")
class StandardizeMetadata(Standardize):
    def __init__(self, max_counts=np.inf, epsilon=1e-6, **kwargs):
        kwargs.pop('center', None)
        super().__init__(center=True, max_counts=max_counts, epsilon=epsilon, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop('center')
        return config

    def call(self, inputs, training=None):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        metadata_out = super().call(metadata, training=training)

        return (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata_out,
            iobs,
            sigiobs,
        )

@tfk.saving.register_keras_serializable(package="abismal")
class StandardizeIntensities(Standardize):
    def __init__(self, max_counts=np.inf, epsilon=1e-6, **kwargs):
        kwargs.pop('center', None)
        super().__init__(center=False, max_counts=max_counts, epsilon=epsilon, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop('center')
        return config

    def call(self, inputs, training=None):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs
        w = 1.
        #w = tf.math.reciprocal(tf.math.square(sigiobs))
        #w = w / tf.reduce_mean(w)

        iout = super().call(w * iobs, training=training)
        sigout = self.standardize(w * sigiobs)
        m = tf.squeeze(self.mean)
        self.add_metric(m, "Imean")
        s = tf.squeeze(self.std)
        self.add_metric(s, "Istd")

        return (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iout,
            sigout,
        )




if __name__=="__main__":
    l = 15
    d = 5
    batches = 5
    x = np.exp(np.random.normal(0., 1., size=(l, d)))
    w = np.exp(np.random.normal(0., 1., size=(l, d)))
    #x = np.arange(l)[...,None] * np.ones((l, d))
    x = x.astype('float32')
    n = RunningMoments()
    s = Standardize()
    for batch in np.split(x, batches):
        n.update(batch)
        s(batch)
        assert np.allclose(n.mean, s.mean)
        assert np.allclose(n.var, s.var)
        assert np.allclose(n.std, s.std)
    assert np.allclose(n.mean, x.mean(-2, keepdims=True))
    assert np.allclose(n.std, x.std(-2, keepdims=True))

    #Test weighted mean
  
    n = RunningMoments()
    s = Standardize()
    for batch,weights in zip(np.split(x, batches), np.split(w, batches)):
        n.update(batch, weights=weights)
    weighted_mean = np.average(x, axis=-2, weights=w, keepdims=True)
    weighted_variance = np.average(np.square(x - weighted_mean), axis=-2, weights=w, keepdims=True)
    weighted_stddev = np.sqrt(weighted_variance)

    assert np.allclose(n.mean, weighted_mean)
    assert np.allclose(n.std, weighted_stddev)
