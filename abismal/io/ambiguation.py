from abismal.symmetry import Op
from reciprocalspaceship.decorators import spacegroupify,cellify
import gemmi
import numpy as np

class Ambiguator():
    def __init__(self, ops):
        self.ops = [Op(op) for op in ops]

    @classmethod
    @spacegroupify
    @cellify
    def from_symmetry(cls, cell, spacegroup):
        reindexing_ops = ["x,y,z"]
        ops = gemmi.find_twin_laws(cell, spacegroup, 3.0, False)
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops]
        return cls(reindexing_ops)

    def ambiguate(self, hkl):
        op = np.random.choice(self.ops)
        hkl = op(hkl)
        return hkl

    def __call__(self, X, Y):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = X
        hkl = self.ambiguate(hkl_in)
        X = (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) 
        return (X, Y)

