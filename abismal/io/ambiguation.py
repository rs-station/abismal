from abismal.symmetry import Op
from reciprocalspaceship.decorators import spacegroupify,cellify
import gemmi
import tensorflow as tf

class Ambiguator():
    def __init__(self, ops, seed=4321):
        self.ops = [Op(op) for op in ops]
        self.seed = seed

    @classmethod
    @spacegroupify
    @cellify
    def from_symmetry(cls, cell, spacegroup):
        reindexing_ops = ["x,y,z"]
        ops = gemmi.find_twin_laws(cell, spacegroup, 3.0, False)
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops]
        return cls(reindexing_ops)

    def ambiguate(self, hkl, index):
        """ Deterministically assign one reindexing op to the image at `index` """
        if len(self.ops) == 1:
            return hkl
        candidates = tf.stack([op(hkl) for op in self.ops], axis=0)
        i = tf.random.stateless_uniform(
            [], (index + self.seed, index + self.seed),
            minval=0, maxval=len(self.ops), dtype=tf.int32,
        )
        return tf.gather(candidates, i)

    def __call__(self, index, XY):
        X,Y = XY
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = X
        hkl = self.ambiguate(hkl_in, index)
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

