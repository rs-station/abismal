from reciprocalspaceship.decorators import spacegroupify,cellify
import gemmi
import numpy as np
from abismal.io.utils import index_uniform


def reindex(hkl, op):
    """Apply a gemmi.Op's rotation to integer miller indices `hkl` [n,3]."""
    if op == gemmi.Op("x,y,z"):
        return hkl
    rot = np.array(op.rot, dtype=np.int64)
    out = np.floor_divide(hkl.astype(np.int64) @ rot, op.DEN)
    return out.astype(hkl.dtype)


class Ambiguator():
    """Reindex each image by one of a set of operators, chosen per image.

    The choice is a fixed function of the image's position in the dataset and
    the seed, so an image keeps the same indexing on every pass and every run.
    """
    def __init__(self, ops, seed=4321):
        self.ops = [gemmi.Op(op) if isinstance(op, str) else op for op in ops]
        self.seed = seed

    @classmethod
    @spacegroupify
    @cellify
    def from_symmetry(cls, cell, spacegroup):
        reindexing_ops = ["x,y,z"]
        ops = gemmi.find_twin_laws(cell, spacegroup, 3.0, False)
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops]
        return cls(reindexing_ops)

    def choose(self, index):
        """ The index into `self.ops` used for the image at `index` """
        return int(index_uniform(index, self.seed) * len(self.ops))

    def ambiguate(self, hkl, index):
        """ Deterministically assign one reindexing op to the image at `index` """
        if len(self.ops) == 1:
            return hkl
        return reindex(hkl, self.ops[self.choose(index)])

    def __call__(self, index, record):
        """ Reindex the hkl of one image record (see `DataLoader`) """
        record = tuple(record)
        hkl = self.ambiguate(record[1], index)
        return record[:1] + (hkl,) + record[2:]
