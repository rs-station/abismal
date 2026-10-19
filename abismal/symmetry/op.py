import gemmi
import numpy as np


class Op:
    """
    A reindexing operator acting on integer miller indices, given as a gemmi
    triplet such as "-x,-y,z".

    Works on numpy arrays and on torch tensors alike. Integer arithmetic rather
    than a float matmul: CUDA has no integer matmul, and a float32 product on an
    Ampere card runs through TF32, whose 10-bit mantissa cannot be trusted to
    round-trip a miller index.
    """
    def __init__(self, triplet):
        self.gemmi_op = gemmi.Op(triplet)
        self.rot = np.array(self.gemmi_op.rot, dtype=np.int64)
        self.den = int(self.gemmi_op.DEN)
        self.identity = self.gemmi_op == 'x,y,z'
        self._torch_rot = {}

    def __str__(self):
        return f"Op({self.gemmi_op.triplet()})"

    def triplet(self):
        return self.gemmi_op.triplet()

    def _rot_like(self, hkl):
        if isinstance(hkl, np.ndarray):
            return self.rot
        key = (hkl.device, hkl.dtype)
        if key not in self._torch_rot:
            import torch
            self._torch_rot[key] = torch.as_tensor(self.rot, dtype=hkl.dtype, device=hkl.device)
        return self._torch_rot[key]

    def __call__(self, hkl):
        if self.identity:
            return hkl
        rot = self._rot_like(hkl)
        # hkl @ rot, written as a broadcast sum so it stays in integers
        out = (hkl[..., :, None] * rot).sum(-2)
        return out // self.den



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
