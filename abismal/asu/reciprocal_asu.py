import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from tensorflow import keras as tfk
from reciprocalspaceship.utils import hkl_to_asu,is_absent
from functools import wraps
from inspect import signature
from reciprocalspaceship.decorators import spacegroupify,cellify

class ReciprocalASUCollection(object):
    def __init__(self, *reciprocal_asus):
        self.reciprocal_asus = reciprocal_asus

    def gather(self, tensors, asu_id, hkl):
        out = None
        for i,(tensor, rasu) in enumerate(zip(tensors, self.reciprocal_asus)):
            miller_ids = rasu._miller_ids(hkl)
            aidx = (miller_ids >= 0) & (asu_id == i)
            miller_ids = tf.maximum(miller_ids, 0)
            vals = tf.gather(tensor, miller_ids)
            if out is None:
                out = vals
            else:
                out = tf.where(
                    aidx, 
                    vals,
                    out,
                )
        return out

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

    def _ensure_in_range(self, H):
        """ Cast any out of bounds indices to [0,0,0] """
        idx = tf.reduce_any(tf.abs(H) > self.hmax, axis=-1, keepdims=True)
        out = tf.where(
            idx,
            tf.zeros_like(H),
            H
        )
        return out

    def _wrap_negative_millers(self, H):
        """ Wrap negative miller indices to positive for gather """
        return tf.where(H >= 0, H, H + self.grid_size[...,:])

    def _sanitize_millers(self, H):
        H = self._ensure_in_range(H)
        H = self._wrap_negative_millers(H)
        return H

    def _miller_ids(self, H):
        H = tf.cast(H, 'int32')
        hplus = self._sanitize_millers(H)
        return tf.gather_nd(self.miller_id, hplus)

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

