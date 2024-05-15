import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
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
            activation='relu',
            seed=1234,
            dropout=None, 
            hidden_units=None, 
            **kwargs,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.kernel_initializer = kernel_initializer
        self.seed=seed
        self.ff = FeedForward(
            kernel_initializer=kernel_initializer, 
            activation=activation,
            dropout=dropout,
            hidden_units=hidden_units,
        )

    def build(self, shapes):

        dmodel = shapes[-1]
        fan_avg = 0.5 * (self.key_dim + dmodel)
        scale = tf.sqrt(0.1 / fan_avg / self.num_heads)
        kernel_initializer = tfk.initializers.TruncatedNormal(stddev=scale, seed=self.seed)
        
        self.key_dense = tfk.layers.EinsumDense(
            "...nd,hdk->...hnk",
            (self.num_heads, None, self.key_dim),
            kernel_initializer=kernel_initializer,
        )
        #self.value_dense = tfk.layers.EinsumDense(
        #    "...nd,hdk->...hnk",
        #    (self.num_heads, None, self.key_dim),
        #    kernel_initializer=self.kernel_initializer,
        #)

        fan_avg = dmodel
        scale = tf.sqrt(0.1 / fan_avg / self.num_heads)
        kernel_initializer = tfk.initializers.TruncatedNormal(stddev=scale, seed=self.seed)
        self.output_dense = tfk.layers.EinsumDense(
            "...hnv,hdv->...nd",
            (None, dmodel),
            kernel_initializer=kernel_initializer,
        )

    @staticmethod
    def sample_ragged_dim(ragged, length):
        """
        Randomly subsample "length" entries from ragged with replacement.
        """
        logits = tf.where(
            tf.ones_like(ragged[...,0], dtype='bool').to_tensor(),
            1.,
            -np.inf,
        )
        idx2 = tf.random.categorical(
            logits,
            length,
            dtype='int32',
        )
        batch_shape = tfk.backend.shape(idx2)[0]
        idx1 = tf.range(batch_shape)[:,None] * tf.ones_like(idx2)
        idx = tf.stack((idx1, idx2), axis=-1)
        out = tf.gather_nd(ragged, idx)
        return out

    @staticmethod
    def to_dense(ragged : tf.RaggedTensor) -> (tf.Tensor, tf.Tensor):
        mask = tf.ones_like(ragged[...,0])
        mask_1d = mask.to_tensor()
        mask = tf.einsum("...a,...b->...ab", mask_1d, mask_1d)
        mask = mask + (1. - mask_1d[...,None])
        mask = tf.cast(mask, 'bool')
        dense = ragged.to_tensor()
        return dense, mask

    def call(self, data, attention_mask=None, **kwargs):
        v = data
        k = self.key_dense(v)
        v = data #self.value_dense(v)

        scores = tf.einsum("...nk,...mk->...nm", k, k)

        if attention_mask is not None:
            scores = tf.where(attention_mask[...,None,:,:], scores, -np.inf)

        scores = tf.nn.softmax(scores, axis=-1)

        out = tf.einsum("...hnm,...md->...hnd", scores, v)
        out = tf.nn.relu(out)
        out = self.output_dense(out)
        out = out + data
        out = self.ff(out)

        #out = tf.concat(tf.unstack(out, axis=-3), axis=-1)

        return out

class TransformerBlock(tfk.layers.Layer):
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

    def call(self, data, attention_mask=None, **kwargs):
        out = self.attention(data, attention_mask=attention_mask) 
        out = self.ff(out) 
        return out

class TransformerAverage(tfk.layers.Layer):
    """
    Transformer followed by average. 
    """
    def __init__(
            self,
            depth,
            num_heads=8,
            key_dim=16,
            dropout=None, 
            hidden_units=None, 
            activation='ReLU',
            kernel_initializer='glorot_normal', 
            normalize=False, 
            max_length=96,
            **kwargs
        ):
        super().__init__()
        self.blocks = []
        self.blocks.append(
            FeedForward(
                hidden_units=None, 
                activation='ReLU',
                kernel_initializer='glorot_normal', 
                normalize=normalize,
            )
        )
        self.max_length=96
        for i in range(depth):
            self.blocks.append(
                MultiHeadAttention(
                    num_heads=8,
                    key_dim=16,
                    dropout=None, 
                    hidden_units=None, 
                    activation='ReLU',
                    kernel_initializer='glorot_normal', 
                    normalize=normalize, 
                )
            )

    def call(self, data, **kwargs):
        out = MultiHeadAttention.sample_ragged_dim(data, self.max_length)

        for block in self.blocks:
            out = block(out)

        out = tf.reduce_mean(out, axis=-2, keepdims=True)

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

