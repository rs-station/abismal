import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.layers import FeedForward


class MultiHeadAttention(tfk.layers.Layer):
    """
    A less pathological attention layer
    """
    #TODO: add dropout support
    def __init__(
            self,
            num_heads,
            key_dim,
            kernel_initializer='glorot_normal',
            **kwargs,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.kernel_initializer = kernel_initializer

    def build(self, shapes):
        self.key_dense = tf.keras.layers.EinsumDense(
            "...nd,hdk->...hnk",
            (self.num_heads, None, self.key_dim),
        )
        self.output_dense = tfk.layers.Dense(shapes[-1], kernel_initializer=self.kernel_initializer)

    def call(self, data, attention_mask=None, **kwargs):
        v = data
        mask = None
        if isinstance(data, tf.RaggedTensor):
            v = v.to_tensor()
            mask = tf.ones_like(data[...,0])
            mask_1d = mask.to_tensor()
            mask = tf.einsum("...a,...b->...ab", mask_1d, mask_1d)
            mask = mask + (1. - mask_1d[...,None])
            mask = tf.cast(mask, 'bool')

        k = self.key_dense(v)

        scores = tf.einsum("...nk,...mk->...nm", k, k)
        scores = tf.where(mask[...,None,:,:], scores, -np.inf)
        scores = tf.nn.softmax(scores, axis=-1)

        out = tf.einsum("abcd,ace->acbe", scores, v)
        out = tf.concat(tf.unstack(out, axis=-2), axis=-1)

        out = self.output_dense(out)
        if isinstance(data, tf.RaggedTensor):
            out = tf.RaggedTensor.from_tensor(out, lengths=data.row_lengths(), row_splits_dtype=data.row_splits.dtype)

        return out

def ragged_to_dense(tensor):
    """ Convert a ragged tensor to dense with an attention mask """
    v = v.to_tensor()
    mask = tf.ones_like(data[...,0])
    mask_1d = mask.to_tensor()
    mask = tf.einsum("...a,...b->...ab", mask_1d, mask_1d)
    mask = mask + (1. - mask_1d[...,None])
    mask = tf.cast(mask, 'bool')

class Transformer(tfk.layers.Layer):
    """
    Transformer self-attention block
    """
    def __init__(
            self,
            num_heads=8,
            key_dim=16,
            dropout=None, 
            hidden_units=None, 
            activation='ReLU',
            kernel_initializer='glorot_normal', 
            normalize=False, 
            **kwargs
        ):
        super().__init__()
        self.ff = FeedForward(
            dropout=dropout, 
            hidden_units=hidden_units, 
            activation=activation, 
            kernel_initializer=kernel_initializer, 
            normalize=normalize
        )
        self.attention = MultiHeadAttention(
            num_heads,
            key_dim,
            kernel_initializer=kernel_initializer,
        )

    def call(self, data, **kwargs):
        out = self.attention(data) + data
        out = self.ff(out) 
        return out


if __name__=="__main__":
    from pylab import *

    n = 200
    vals = np.random.random((n, 6))
    idx = np.sort(np.random.choice(10, size=n))
    x = tf.RaggedTensor.from_value_rowids(vals, idx)

    n = MultiHeadAttention(8, 16)
    y = n(x)
    print(y.shape)

    n = Transformer(8, 16)
    y = n(x)
    print(y.shape)

