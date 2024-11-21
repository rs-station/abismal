import numpy as np
from reciprocalspaceship.utils import hkl_to_asu,is_absent
from reciprocalspaceship.decorators import spacegroupify,cellify
import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from inspect import signature
import gemmi
from abismal.symmetry.op import Op

@tfk.saving.register_keras_serializable(package="abismal")
class ReciprocalASUCollection(tfk.layers.Layer):
    def __init__(self, *reciprocal_asus, **kwargs):
        super().__init__(**kwargs)
        self.reciprocal_asus = reciprocal_asus
        asu_ids,dHKL,centric,epsilon = [],[],[],[]
        Hunique = []
        self.asu_size = 0
        self.hmax = 0
        for asu_id, rasu in enumerate(reciprocal_asus):
            dHKL.append(rasu.dHKL)
            centric.append(rasu.centric)
            epsilon.append(rasu.epsilon)
            self.asu_size = self.asu_size + rasu.asu_size
            self.hmax = np.maximum(self.hmax, rasu.hmax)
            asu_ids.append(tf.ones(rasu.asu_size, dtype='int32') * asu_id)
            Hunique.append(rasu.Hunique)

        self.centric = tf.concat(centric, axis=-1)
        self.epsilon = tf.concat(epsilon, axis=-1)
        self.dHKL = tf.concat(dHKL, axis=-1)
        self.asu_id = tf.concat(asu_ids, axis=-1)
        self.Hunique = tf.concat(Hunique, axis=-2)

        h,k,l = self.hmax
        self.miller_id = -np.ones((len(self), 2*h+1, 2*k+1, 2*l+1), dtype=np.int32)
        self.grid_size = tf.convert_to_tensor((2*h+1, 2*k+1, 2*l+1), dtype=tf.int32)
        offset = 0
        for asu_id, rasu in enumerate(reciprocal_asus):
            h,k,l = rasu.Hall.T
            self.miller_id[asu_id *np.ones_like(h),h,k,l] = rasu.miller_id[h,k,l] + offset
            offset = offset + rasu.asu_size
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {f'rasu_{i}' : tfk.saving.serialize_keras_object(v) for i,v in enumerate(self.reciprocal_asus)}
        )
        return config

    @classmethod
    def from_config(cls, config):
        rasus = []
        keys = config.keys()
        rasus = [tfk.saving.deserialize_keras_object(v) for k,v in config.items() if k.startswith('rasu_')]
        config = {k:v for k,v in config.items() if not k.startswith('rasu_')}
        return cls(*rasus, **config)

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

    def _miller_ids(self, asu_id, H):
        H = tf.cast(H, 'int32')
        asu_id = tf.cast(asu_id, 'int32')
        hplus = self._sanitize_millers(H)
        ahkl = tf.concat((asu_id, hplus), axis=-1)
        return tf.gather_nd(self.miller_id, ahkl)

    def _gather(self, tensor, asu_id, H):
        idx = self._miller_ids(asu_id, H)
        safe = tf.maximum(idx, 0)
        return tf.gather(tensor, safe)

    def gather(self, tensor, asu_id, H):
        """
        Parameters
        ----------
        tensor : tf.Tensor 
            A tensor in the length of self.asu_size
        asu_id : tf.Tensor
            A potentially ragged tensor of asu ids
        H : tf.Tensor
            A potentially ragged tensor of miller indices
        """
        return tf.ragged.map_flat_values(self._gather, tensor, asu_id, H)

    def __iter__(self):
        return self.reciprocal_asus.__iter__()

    def __next__(self):
        return self.reciprocal_asus.__next__()

    def __len__(self):
        return len(self.reciprocal_asus)

@tfk.saving.register_keras_serializable(package="abismal")
class ReciprocalASU(tfk.layers.Layer):
    @cellify
    @spacegroupify
    def __init__(self, cell, spacegroup, dmin, anomalous=True, **kwargs):
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
        super().__init__(**kwargs)
        self.anomalous = anomalous
        self._cell = cell.parameters
        self._spacegroup = spacegroup.xhm()
        self.dmin = dmin
        go = self.spacegroup.operations()

        H = self.get_reciprocal_cell()

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
        h,k,l = self.hmax
        self.miller_id = -np.ones((2*h+1, 2*k+1, 2*l+1), dtype=np.int32)
        self.miller_id[H[:,0], H[:,1], H[:,2]] = np.arange(self.asu_size)[inv]
        self.Hall = H

        self.dHKL = cell.calculate_d_array(self.Hunique).astype(np.float32)
        self.epsilon = go.epsilon_factor_array(self.Hunique).astype(np.float32)
        self.centric = go.centric_flag_array(self.Hunique).astype(bool)
        self.built = True #This object is not lazy

    def get_config(self):
        config = super().get_config()
        config.update({
            'cell' : self.cell.parameters,
            'spacegroup' : self.spacegroup.xhm(),
            'dmin' : self.dmin,
            'anomalous' : self.anomalous,
        })
        return config

    @property
    def spacegroup(self):
        return gemmi.SpaceGroup(self._spacegroup)

    @property
    def cell(self):
        return gemmi.UnitCell(*self._cell)

    def get_reciprocal_cell(self):
        """ Generate the full reciprocal cell respecting systematic absences """
        go = self.spacegroup.operations()
        self.hmax = np.array(self.cell.get_hkl_limits(self.dmin))

        h,k,l = self.hmax

        self.grid_size = tf.convert_to_tensor((2*h+1, 2*k+1, 2*l+1), dtype=tf.int32)
        H = np.mgrid[-h:h+1:1,-k:k+1:1,-l:l+1:1].reshape((3, -1)).T

        #Apply resolution cutoff and remove absences
        d = self.cell.calculate_d_array(H)
        H = H[d >= self.dmin]
        H = H[~is_absent(H, self.spacegroup)]

        #Remove 0, 0, 0
        H = H[np.any(H != 0, axis=1)]

        return H

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
        idx = self._miller_ids(H)
        safe = tf.maximum(idx, 0)
        return tf.gather(tensor, safe)

@tfk.saving.register_keras_serializable(package="abismal")
class ReciprocalASUGraph(ReciprocalASUCollection):
    def __init__(self, *reciprocal_asus, parents=None, reindexing_ops=None, **kwargs):
        super().__init__(*reciprocal_asus, **kwargs)
        if parents is None:
            parents = tf.range(len(reciprocal_asus))
        self.parents = parents
        self.reindexing_ops = reindexing_ops
        self.parent_asu_id = tf.gather(parents, self.asu_id)
        Hpa = []
        is_root = []
        for i,rasu in enumerate(self.reciprocal_asus):
            op = 'x,y,z'
            if reindexing_ops is not None:
                op = reindexing_ops[i]
            op = Op(op)
            Hpa.append(op(rasu.Hunique))

        Hpa = tf.concat(Hpa, axis=0)
        self.parent_hkl = Hpa
        self.is_root = self.parent_asu_id == self.asu_id
        self.parent_miller_id = self._miller_ids(self.parent_asu_id[:,None], self.parent_hkl)
        self.has_parent = (self.parent_miller_id >= 0) & (~self.is_root)

    @classmethod
    def from_lists(
        cls,
        cells,
        spacegroups,
        dmins,
        anomalous,
        parents=None,
        reindexing_ops=None,
    ):
        rasu = []
        for i, (cell, sg, dmin, anom) in enumerate(zip(cells, spacegroups, dmins, anomalous)):
            rasu.append(
                ReciprocalASUCollection(cell, sg, dmin, anom)
            )
        return cls(rasu, parents, reindexing_ops)

    def get_config(self):
        config = super().get_config()
        config.update({
            'parents' : [int(i) for i in self.parents],
            'reindexing_ops' : self.reindexing_ops,
        })
        return config


