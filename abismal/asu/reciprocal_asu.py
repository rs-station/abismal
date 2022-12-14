import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from tensorflow import keras
from reciprocalspaceship.utils import hkl_to_asu,is_absent
from functools import wraps
from inspect import signature
from reciprocalspaceship.decorators import spacegroupify,cellify

class ReciprocalASUCollection(object):
    def __init__(self, *reciprocal_asus):
        self.reciprocal_asus = reciprocal_asus

    def gather(self, asu_id, hkl):
        pass

class ReciprocalASU(object):
    @cellify
    @spacegroupify
    def __init__(self, cell, spacegroup, dmin, anomalous=True):
        """
        Base Layer that remaps observed miller indices to the reciprocal asu.
        This class enables indexing of per reflection variables from tf.
        All the attributes are in the length of the number of non-redudant 
        miller indices.

        Attributes
        ----------
        asu_size : int
            Number of non-redundant reflections in the ASU
        dHKL : np.ndarray(np.float32)
            Reflection resolution
        centric : np.ndarray(bool)
            Indicator for centric vs acentric reflections
        epsilon : np.ndarray(np.float32)
            Reflection multiplicity. 

        Parameters
        ----------
        cell : gemmi.UnitCell
            A gemmi.UnitCell object
        spacegroup : gemmi.SpaceGroup
            A gemmi SpaceGroup object
        dmin : float
            The maximum resolution in Ångstroms
        anomalous : bool
            If true, treat Friedel mates as non-redudant
        """
        self.anomalous = anomalous
        self.cell = cell
        self.spacegroup = spacegroup
        go = spacegroup.operations()
        self.hmax = np.array(cell.get_hkl_limits(dmin))
        self.dmin = dmin
        h,k,l = self.hmax

        self.grid_size = tf.convert_to_tensor((2*h+1, 2*k+1, 2*l+1), dtype=tf.int32)
        H = np.mgrid[-h:h+1:1,-k:k+1:1,-l:l+1:1].reshape((3, -1)).T

        #Apply resolution cutoff and remove absences
        d = cell.calculate_d_array(H)
        H = H[d >= dmin]
        H = H[~is_absent(H, spacegroup)]

        #Remove 0, 0, 0
        H = H[np.any(H != 0, axis=1)]

        Hasu,Isym = hkl_to_asu(H, spacegroup)
        if anomalous:
            friedel_sign = np.array([-1, 1])[Isym % 2][:,None]
            centric = go.centric_flag_array(Hasu).astype(bool)
            friedel_sign[centric] = 1
            Hasu = friedel_sign*Hasu

        self.Hunique,inv = np.unique(Hasu, axis=0, return_inverse=True)
        self.asu_size = len(self.Hunique)

        #This 3d grid contains the unique miller id for each reflection
        #-1 means that you shouldn't observe this miller index in the
        #data set
        self.miller_id = -np.ones((2*h+1, 2*k+1, 2*l+1), dtype=np.int32)
        self.miller_id[H[:,0], H[:,1], H[:,2]] = np.arange(self.asu_size)[inv]

        self.dHKL = cell.calculate_d_array(self.Hunique).astype(np.float32)
        self.epsilon = go.epsilon_factor_array(self.Hunique).astype(np.float32)
        self.centric = go.centric_flag_array(self.Hunique).astype(bool)

    @classmethod
    def from_reference_data(cls, mtz_filename, dmin=None, anomalous=True):
        import reciprocalspaceship as rs
        ds = rs.read_mtz(mtz_filename)
        if anomalous:
            ds = ds.stack_anomalous()
        ds = ds.dropna().compute_dHKL()
        if dmin is None:
            dmin = ds.dHKL.min()
        ds = ds[ds.dHKL >= dmin]
        Hunique = ds.get_hkls().astype('int64')
        return cls(ds.cell, ds.spacegroup, dmin, anomalous, Hunique)

    def _miller_ids(self, H):
        return tf.gather_nd(self.miller_id, tf.where(H >= 0, H, H + self.grid_size[...,:]))

    def gather(self, tensor, H):
        """
        Parameters
        ----------
        tensor : tf.Tensor 
            A tensor in the length of self.asu_size
        H : tf.Tensor
            A potentially ragged tensor of miller indices
        """
        return tf.gather(tensor, self._miller_ids(H))

if __name__=="__main__":
    import gemmi
    from IPython import embed
    sg = gemmi.SpaceGroup(19)
    cell = gemmi.UnitCell(10., 20., 30., 90., 90., 90.)
    anomalous = True
    dmin = 2.
    L = ReciprocalASU(cell, sg, dmin, anomalous)
    h,k,l = L.hmax
    H = np.mgrid[-h:h+1:1,-k:k+1:1,-l:l+1:1].reshape((3, -1)).T

    #Apply resolution cutoff and remove absences
    d = cell.calculate_d_array(H)
    H = H[d >= dmin]
    H = H[~is_absent(H, sg)]

    Hasu,Isym = hkl_to_asu(H, sg)
    if anomalous:
        Hasu = np.array([-1, 1])[Isym % 2][:,None]*Hasu

    Hunique,inv = np.unique(Hasu, axis=0, return_inverse=True)
    miller_ids = np.arange(inv.max() + 1)[inv].astype(np.float32)
    
    #Check broadcasting for single 
    assert L._miller_ids(H[0]) == miller_ids[0]

    #Check broadcasting for multiple
    assert np.array_equal(L._miller_ids(H).numpy() , miller_ids)

    #Check broadcasting for batched data
    from tqdm import trange
    for i in trange(100):
        size = 10000
        H = H[np.random.choice(len(H), size)].astype(np.int32)
        #batch = np.sort(np.random.choice(np.random.choice([-1,0,1])+10, size)).astype(np.int32)
        batch = np.sort(np.random.choice(10, size)).astype(np.int32)
        R = tf.RaggedTensor.from_value_rowids(H, batch)
        ids = L._miller_ids(R)

    embed(colors='Linux')
