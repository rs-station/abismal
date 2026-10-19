import numpy as np
import torch
from reciprocalspaceship.utils import hkl_to_asu,is_absent
from reciprocalspaceship.decorators import spacegroupify,cellify
import gemmi
from abismal.symmetry.op import Op


def _flat_miller_index(asu_id, H, hmax):
    """
    Position of (asu_id, h, k, l) in the flattened, per-asu miller id grid.

    Any reflection with a component past `hmax` is mapped to (0, 0, 0) first,
    which is never in an ASU, so it reads back as id -1. Negative indices wrap
    to the top of the grid the way numpy would index them.
    """
    h, k, l = (int(i) for i in hmax)
    shape = (2*h + 1, 2*k + 1, 2*l + 1)
    if isinstance(H, np.ndarray):
        H = H.astype(np.int64)
        out_of_range = (np.abs(H) > np.array(hmax)).any(-1, keepdims=True)
        H = np.where(out_of_range, 0, H)
        H = np.where(H >= 0, H, H + np.array(shape))
        asu_id = np.asarray(asu_id, dtype=np.int64)
    else:
        H = H.long()
        bound = H.new_tensor(hmax)
        out_of_range = (H.abs() > bound).any(-1, keepdim=True)
        H = torch.where(out_of_range, 0, H)
        H = torch.where(H >= 0, H, H + H.new_tensor(shape))
        asu_id = asu_id.long()
    a = asu_id[..., 0]
    return ((a * shape[0] + H[..., 0]) * shape[1] + H[..., 1]) * shape[2] + H[..., 2]
class ReciprocalASU:
    @cellify
    @spacegroupify
    def __init__(self, cell, spacegroup, dmin, anomalous=True):
        """
        Enumerates the non-redundant miller indices of one reciprocal asymmetric
        unit and maps observed miller indices onto them. Everything here is
        numpy; `ReciprocalASUCollection` moves what the model needs onto the
        device.

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
        miller_id : np.ndarray(np.int32)
            A (2h+1, 2k+1, 2l+1) grid holding the index into Hunique of every
            miller index, or -1 for those that are not in the data set.

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
        inv = inv.reshape(-1)
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

    @property
    def spacegroup(self):
        return gemmi.SpaceGroup(self._spacegroup)

    @property
    def cell(self):
        return gemmi.UnitCell(*self._cell)

    def get_reciprocal_cell(self):
        """ Generate the full reciprocal cell respecting systematic absences """
        self.hmax = np.array(self.cell.get_hkl_limits(self.dmin))

        h,k,l = self.hmax

        self.grid_size = np.array((2*h+1, 2*k+1, 2*l+1), dtype=np.int32)
        H = np.mgrid[-h:h+1:1,-k:k+1:1,-l:l+1:1].reshape((3, -1)).T

        #Apply resolution cutoff and remove absences
        d = self.cell.calculate_d_array(H)
        H = H[d >= self.dmin]
        H = H[~is_absent(H, self.spacegroup)]

        #Remove 0, 0, 0
        H = H[np.any(H != 0, axis=1)]

        return H

    def miller_ids(self, H):
        """Index into Hunique of each miller index in `H`, -1 if absent."""
        idx = _flat_miller_index(np.zeros(H.shape[:-1] + (1,), dtype=np.int64), np.asarray(H), self.hmax)
        return self.miller_id.reshape(-1)[idx]

    def gather(self, tensor, H):
        """
        Parameters
        ----------
        tensor : np.ndarray
            An array in the length of self.asu_size
        H : np.ndarray
            Miller indices, shape (..., 3)
        """
        idx = self.miller_ids(H)
        safe = np.maximum(idx, 0)
        return np.asarray(tensor)[safe]
class ReciprocalASUCollection(torch.nn.Module):
    def __init__(self, *reciprocal_asus):
        """
        Several reciprocal ASUs concatenated into one flat index space, which is
        how the surrogate posterior lays out its parameters.

        The lookup tables are non-persistent buffers: they follow the module to
        the device but stay out of the state_dict, because they are rebuilt
        exactly from the config.
        """
        super().__init__()
        self.reciprocal_asus = reciprocal_asus
        asu_ids,dHKL,centric,epsilon = [],[],[],[]
        Hunique = []
        self.asu_size = 0
        hmax = np.zeros(3, dtype=int)
        for asu_id, rasu in enumerate(reciprocal_asus):
            dHKL.append(rasu.dHKL)
            centric.append(rasu.centric)
            epsilon.append(rasu.epsilon)
            self.asu_size = self.asu_size + rasu.asu_size
            hmax = np.maximum(hmax, rasu.hmax)
            asu_ids.append(np.full(rasu.asu_size, asu_id, dtype=np.int64))
            Hunique.append(rasu.Hunique)
        self.hmax = tuple(int(i) for i in hmax)

        self._buffer('centric', np.concatenate(centric, axis=-1))
        self._buffer('epsilon', np.concatenate(epsilon, axis=-1))
        self._buffer('dHKL', np.concatenate(dHKL, axis=-1))
        self._buffer('asu_id', np.concatenate(asu_ids, axis=-1))
        self._buffer('Hunique', np.concatenate(Hunique, axis=-2).astype(np.int64))

        h,k,l = self.hmax
        miller_id = -np.ones((len(self), 2*h+1, 2*k+1, 2*l+1), dtype=np.int64)
        offset = 0
        for asu_id, rasu in enumerate(reciprocal_asus):
            h,k,l = rasu.Hall.T
            miller_id[asu_id *np.ones_like(h),h,k,l] = rasu.miller_id[h,k,l] + offset
            offset = offset + rasu.asu_size
        self._buffer('miller_id', miller_id.reshape(-1))

    def _buffer(self, name, value):
        self.register_buffer(name, torch.as_tensor(value), persistent=False)

    def _miller_ids(self, asu_id, H):
        """
        Flat index into the collection of each (asu_id, H), or -1 when that
        reflection is not in the collection.

        Parameters
        ----------
        asu_id : torch.Tensor
            Integer asu ids, shape (..., 1)
        H : torch.Tensor
            Miller indices, shape (..., 3)

        Returns
        -------
        Integer tensor of shape (...,)
        """
        return self.miller_id[_flat_miller_index(asu_id, H, self.hmax)]

    def _gather(self, tensor, asu_id, H):
        idx = self._miller_ids(asu_id, H)
        safe = idx.clamp_min(0)
        return tensor[safe]

    def valid(self, asu_id, H):
        """Which of these miller indices does the collection actually contain?

        `gather` has nothing to return for an index outside the ASU, so it
        falls back on index 0. That is a silent lie -- the reflection gets the
        posterior of whatever happens to sit first in the ASU -- and callers
        need to be able to drop those rather than score them. A reindexing
        operator that moves reflections in resolution is the way it happens in
        practice: the outermost images land past dmin.

        Parameters
        ----------
        asu_id : torch.Tensor
            Integer asu ids, shape (..., 1)
        H : torch.Tensor
            Miller indices, shape (..., 3)

        Returns
        -------
        Boolean tensor shaped like `asu_id`.
        """
        return (self._miller_ids(asu_id, H) >= 0)[..., None]

    def gather(self, tensor, asu_id, H):
        """
        Parameters
        ----------
        tensor : torch.Tensor
            A tensor in the length of self.asu_size (leading dimension)
        asu_id : torch.Tensor
            Integer asu ids, shape (..., 1)
        H : torch.Tensor
            Miller indices, shape (..., 3)

        Returns
        -------
        `tensor` gathered to shape (..., *tensor.shape[1:])
        """
        return self._gather(tensor, asu_id, H)

    def __iter__(self):
        return self.reciprocal_asus.__iter__()

    def __len__(self):
        return len(self.reciprocal_asus)
class ReciprocalASUGraph(ReciprocalASUCollection):
    def __init__(self, *reciprocal_asus, parents=None, reindexing_ops=None):
        super().__init__(*reciprocal_asus)
        if parents is None:
            parents = list(range(len(reciprocal_asus)))
        self.parents = [int(i) for i in parents]
        self.reindexing_ops = reindexing_ops

        parent_asu_id = torch.as_tensor(self.parents)[self.asu_id]
        Hpa = []
        for i,rasu in enumerate(self.reciprocal_asus):
            op = 'x,y,z'
            if reindexing_ops is not None:
                op = reindexing_ops[i]
            Hpa.append(Op(op)(rasu.Hunique.astype(np.int64)))
        Hpa = torch.as_tensor(np.concatenate(Hpa, axis=0))

        self._buffer('parent_asu_id', parent_asu_id)
        self._buffer('parent_hkl', Hpa)
        self._buffer('is_root', parent_asu_id == self.asu_id)
        self._buffer('parent_miller_id', self._miller_ids(parent_asu_id[:,None], Hpa))
        self._buffer('has_parent', (self.parent_miller_id >= 0) & (~self.is_root))
