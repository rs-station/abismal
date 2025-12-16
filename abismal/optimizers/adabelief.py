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
class AdaBelief(AbismalOptimizer):
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
            delta = gradient - m
            m.assign_add(
                tf.where(
                    nonzero,
                    (1 - self.beta_1) * delta,
                    0.
                )
            )
            v.assign_add(
                tf.where(
                    nonzero,
                    (self.beta_2 - 1.) * v + (1. - self.beta_2) * tf.square(delta),
                    0.
                )
            )
            variable.assign_sub(
                tf.where(
                    nonzero,
                    (m * alpha) / (tf.sqrt(v) + self.epsilon),
                    0.
                )
            )
        else:
            delta = gradient - m
            m.assign_add((1 - self.beta_1) * delta)
            v.assign_add((self.beta_2 - 1.) * v + (1. - self.beta_2) * tf.square(delta))
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

