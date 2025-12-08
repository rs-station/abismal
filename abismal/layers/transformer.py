from abismal.layers.feed_forward import FeedForward,GLUFeedForward
import numpy as np
import tf_keras as tfk
import tensorflow as tf
import math


class MultiHeadAttention(tfk.layers.Layer):
    def __init__(self,num_heads=8, d=5, initializer='glorot_normal'):
        super().__init__()
        self.num_heads = num_heads
        self.d = d
        self.initializer = initializer

    def build(self, shapes):
        dmodel = shapes[-1]
        self.w_K = self.add_weight('wK', 
           (self.d, self.num_heads, dmodel), initializer=self.initializer)
        self.w_Q = self.add_weight('wQ', 
           (self.d, self.num_heads, dmodel), initializer=self.initializer)
        self.w_V = self.add_weight('wV', 
           (dmodel // self.num_heads, self.num_heads, dmodel), initializer=self.initializer)

    def call(self, X, mask=None):
        K = tf.einsum("...khd,...sd->...hsk", self.w_K, X)
        Q = tf.einsum("...khd,...sd->...hsk", self.w_Q, X)
        V = tf.einsum("...vhd,...sd->...hsv", self.w_V, X)
        logits = tf.einsum("...hsk,...hlk->...hsl", Q, K)
        if mask is not None:
            logits = tf.where(mask, logits, -math.inf)
        logits = tf.nn.softmax(logits, axis=-1)
        O = tf.einsum("...hsl,...hsv->...shv", logits, V)
        shape = tfk.backend.shape(X)
        O = tf.reshape(O, shape)
        return O

class Block(tfk.layers.Layer):
    def __init__(self, num_heads=4, d=8, initializer='glorot_normal'):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, d=d, initializer=initializer)
        self.ff = GLUFeedForward(
            kernel_initializer=initializer,
            normalizer='rms',
            activation='swish',
        )

    def call(self, X, mask=None):
        out = self.mha(X, mask)
        out = self.ff(out)
        return out

class Transformer(tfk.layers.Layer):
    def __init__(self, num_blocks=6):
        super().__init__()
        self.blocks = [Block() for i in range(num_blocks)]

    @staticmethod
    def ragged_to_dense_and_mask(X):
        out = X.to_tensor()
        mask = tf.ones_like(X[...,0], dtype=bool).to_tensor(False)
        #mask = (mask[...,None] & mask[...,None,:])
        N = tf.linalg.diag(
            tf.zeros_like(mask),
            padding_value=True,
        )
        mask = (mask[...,None,:] & N)
        mask = mask[...,None,:,:] #add head dim
        return out, mask

    def _call_ragged(self, X, mask=None):
        if mask is None:
            out,mask = Transformer.ragged_to_dense_and_mask(X)
        else:
            out,_ = Transformer.ragged_to_dense_and_mask(X)

        for block in self.blocks:
            out = block(out, mask)
        out = tf.RaggedTensor.from_tensor(out, lengths=X.row_lengths())
        return out

    def call(self, X, mask=None):
        if isinstance(X, tf.RaggedTensor):
            return self._call_ragged(X, mask)
        out = X
        for block in self.blocks:
            out = block(out, mask)
        return out

if __name__=="__main__":
    t = Transformer()

    b = 13
    dmodel =32
    s = 256
    n = b * s


    v = np.random.random((n, dmodel))
    row = np.sort(np.random.choice(b, n))
    x = tf.RaggedTensor.from_value_rowids(v, row)
    y = t(x)
    from IPython import embed;embed(colors='linux')

