import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from abismal.optimizers.base import AbismalOptimizer

import tf_keras as tfk


class WAdam(AbismalOptimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.9,
        epsilon=1e-12,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="WAdam",
        lazy_vars=None,
        **kwargs
    ):
        super().__init__(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.9,
            epsilon=1e-12,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name=name,
            lazy_vars=lazy_vars,
            **kwargs
        )


    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        beta_1_power = tf.pow(beta_1, local_step)
        beta_2_power = tf.pow(beta_2, local_step)
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        g = gradient
        delta = g - m
        if self.lazy_vars is not None and var_key in self.lazy_vars:
            nonzero = (g != 0.)
            m.assign_add(tf.where(
                nonzero, 
                (1. - beta_1) * delta, 
                0.
            ))
            v.assign_add(tf.where(
                nonzero, 
                (beta_2 - 1.) * v + (1. - beta_2) * delta * (g - m), 
                0.
            ))
            variable.assign_sub(tf.where(
                nonzero, 
                (m * alpha) / (tf.sqrt(v) + self.epsilon),
                0.,
            ))
        else:
            m.assign_add((1. - beta_1) * delta)
            v.assign_add((beta_2 - 1.) * v + (1. - beta_2) * delta * (g - m))
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))


