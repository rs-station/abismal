import tensorflow as tf
import tf_keras as tfk
import gemmi

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

