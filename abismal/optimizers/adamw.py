# This is a derivative of work published in the tensorflow_addons package
# This work is modified to ensure compatibility with legacy keras and tensorflow
# probability and can be used without the tfa package installed.
#
# The original source was released under the following Apache License:
# ==============================================================================
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import importlib
import tf_keras as tfk
import tensorflow as tf

from typing import Union, Callable

from abismal.optimizers.base import AbismalOptimizer

adam_optimizer_class = tfk.optimizers.legacy.Adam

@tf.keras.utils.register_keras_serializable(package="abismal")
class AdamW(AbismalOptimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, ICLR 2019).

    The Keras base `Optimizer` already implements *decoupled* weight decay: its
    `_apply_weight_decay` does `variable.assign_sub(variable * wd * lr)` from
    inside `apply_gradients`, independently of the gradient. So for ordinary
    dense variables, `Adam(weight_decay=...)` is already AdamW and this class
    adds nothing.

    What the base implementation gets wrong here is the *lazy* variables. It
    decays every variable on every step, including entries whose gradient was
    zero only because the corresponding reflection was absent from the batch.
    Those entries would be pulled toward zero at a rate governed by how rarely
    they are observed rather than by the objective -- precisely the coupling
    between update frequency and parameter value that the lazy optimizers exist
    to avoid, and worst for the least-observed parameters.

    So lazy variables opt out of the base-class decay via `_use_weight_decay`,
    and instead get it inside `update_step`, masked by the same `nonzero`
    condition that gates the moment updates. A lazy variable is therefore
    decayed once per *update* rather than once per *step*, which keeps decay and
    adaptation on the same clock. Note the consequence: a parameter updated on
    one step in ten decays ten times more slowly in wall-clock terms than a
    dense one at the same `weight_decay`.

    The `weight_decay` default of 0.004 matches `tf_keras.optimizers.AdamW`.
    Setting it to 0.0 or None recovers plain `Adam` exactly.
    """
    def __init__(self, weight_decay=0.004, name="AdamW", **kwargs):
        super().__init__(weight_decay=weight_decay, name=name, **kwargs)

    def _use_weight_decay(self, variable):
        """Exclude lazy variables from the base class's unconditional decay.

        They are decayed inside `update_step` instead, gated on a nonzero
        gradient. Without this they would be decayed twice, and on every step.
        """
        if self.lazy_vars is not None and self._var_key(variable) in self.lazy_vars:
            return False
        return super()._use_weight_decay(variable)

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if self.lazy_vars is not None and var_key in self.lazy_vars:
            nonzero = gradient != 0.
            m.assign_add(
                tf.where(
                    nonzero,
                    (gradient - m) * (1 - self.beta_1),
                    0.
                )
            )
            v.assign_add(
                tf.where(
                    nonzero,
                    (tf.square(gradient) - v) * (1 - self.beta_2),
                    0.
                )
            )
            update = (m * alpha) / (tf.sqrt(v) + self.epsilon)
            if self.weight_decay is not None:
                # Decoupled: uses the pre-update variable value, matching the
                # order the base class applies decay relative to the step.
                wd = tf.cast(self.weight_decay, variable.dtype)
                update = update + variable * wd * lr
            variable.assign_sub(
                tf.where(
                    nonzero,
                    update,
                    0.
                )
            )
        else:
            # Decoupled decay for dense variables is applied by the base class
            # in apply_gradients, so it is deliberately absent here.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
