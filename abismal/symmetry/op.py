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



def merge_dmin(cell, ops, dmin):
    """The resolution the merging ASU must span for `ops` to stay inside it.

    A reindexing operator only maps a reflection onto another reflection of the
    same resolution when it is a symmetry of the *lattice*. Under a pseudo-
    symmetric operator -- the h/k swap of a cell with a very nearly, but not
    quite, equal to b, say -- the image sits at a different d, and for the
    outermost reflections it lands past dmin. Those have no entry in the ASU,
    and a miller id of -1 is not a resolution the caller can recover from.

    Writing 1/d^2 = h^T G h for the reciprocal metric G and h' = M h, the
    resolution ratio d'/d is the Rayleigh quotient of M^T G M with respect to
    G, so the worst contraction over all h is fixed by the largest generalized
    eigenvalue. That bound is tight in the continuum and slightly conservative
    on the lattice, which is the direction we want.

    Returns `dmin` exactly for the identity and for any true lattice symmetry,
    so padding only costs anything when the operator is pseudo-symmetric.

    Parameters
    ----------
    cell : gemmi.UnitCell
    ops : iterable of str or gemmi.Op
        The reindexing operators the model will apply, including the identity.
    dmin : float
        The resolution the merged output is wanted at.

    Returns
    -------
    float
        The resolution to build the reciprocal ASU to, <= dmin.
    """
    import numpy as np

    G = np.array(cell.reciprocal_metric_tensor().as_mat33().tolist())
    worst = 1.0
    for op in ops:
        if not isinstance(op, gemmi.Op):
            op = gemmi.Op(str(op))
        # gemmi applies rotations to miller indices on the right, so the
        # column-convention matrix taking h to h' is the transpose.
        M = np.array(op.rot, dtype=float).T / op.DEN
        lam = np.linalg.eigvals(np.linalg.solve(G, M.T @ G @ M)).real
        worst = min(worst, 1.0 / np.sqrt(lam.max()))
    return dmin * worst
