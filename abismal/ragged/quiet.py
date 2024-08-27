import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor

def quiet_rep(self, *args, **kwargs):
    shape = self.shape
    dtype = self.dtype
    return f"RaggedTensor with shape {shape} and dtype {dtype}"

RaggedTensor.__repr__ = quiet_rep
RaggedTensor.__str__ = quiet_rep

