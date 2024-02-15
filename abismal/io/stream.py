#/usr/bin/env cctbx.python
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf
from .loader import DataLoader
from reciprocalspaceship.decorators import spacegroupify,cellify
from multiprocessing import cpu_count,Pool



#See Rupp Table 5-2
_cell_constraints = {
    'triclinic' : lambda x: x,
    'orthorhombic' : lambda x: [x[0], x[1], x[2], 90., 90., 90.],
    'monoclinic' : lambda x: [x[0], x[1], x[2], 90., x[4], 90.],
    'hexagonal' : lambda x: [0.5*(x[0] + x[1]), 0.5*(x[0] + x[1]), x[2], 90., 90., 120.],
    'rhombohedral' : lambda x: [0.5*(x[0] + x[1]), 0.5*(x[0] + x[1]), x[2], 90., 90., 120.],
    'cubic' : lambda x: [np.mean(x[:3]), np.mean(x[:3]), np.mean(x[:3]), 90., 90., 90.],
    'tetragonal' : lambda x: [0.5*(x[0] + x[1]), 0.5*(x[0] + x[1]), x[2], 90., 90., 90.],
}

# See crystFEL API reference here: https://www.desy.de/~twhite/crystfel/reference/stream_8h.html
_block_markers = {
    "geometry" : ("----- Begin geometry file -----", "----- End geometry file -----"),
    "chunk" : ("----- Begin chunk -----", "----- End chunk -----"),
    "cell" : ("----- Begin unit cell -----", "----- End unit cell -----"),
    "peaks" : ("Peaks from peak search", "End of peak list"),
    "crystal" : ("--- Begin crystal", "--- End crystal"),
    "reflections" : ("Reflections measured after indexing", "End of reflections"),
}


class StreamLoader():
    @cellify
    @spacegroupify
    def __init__(self, stream_file, cell=None, dmin=None, asu_id=0, min_refls=0, spacegroup=None, wavelength=None):
        super().__init__()
        self.signature = None
        self.min_refls = min_refls
        self.cell = cell
        self.lattice_type = self.get_lattice_type(stream_file)
        if self.cell is None:
            self.cell = self.get_average_cell(stream_file)
            self.cell = _cell_constraints[self.lattice_type](self.cell)
            self.cell = gemmi.UnitCell(*self.cell)
        self.asu_id = asu_id
        self.wavelength = wavelength
        self.stream_file = stream_file
        self.metadata_mean  = None
        self.metadata_std   = None
        self.intensity_std  = None
        self.dmin = dmin
        self.spacegroup = spacegroup

    @staticmethod
    def stream_to_lines(stream_file, encoding='us-ascii'):
        if stream_file.endswith("bz2"):
            import bz2
            for line in bz2.open(stream_file):
                yield line.decode(encoding)
        else:
            for line in open(stream_file):
                yield line

    def get_metadata_dims(self):
        for x,y in self.get_dataset():
            break
        return x[4].shape[-1]

    @staticmethod
    def to_crystals(stream_file):
        return StreamLoader.to_blocks(stream_file, 'crystal')

    @staticmethod
    def get_lattice_type(stream_file):
        for line in StreamLoader.stream_to_lines(stream_file):
            if line.startswith("lattice_type ="):
                lattice_type = line.split()[2]
                return lattice_type
        raise ValueError("No lattice_type entry!")

    @staticmethod
    def get_target_cell(stream_file):
        lines = next(StreamLoader.to_blocks(stream_file, 'cell'))
        cell = [0.]*6
        for line in lines:
            if line.startswith("a ="):
                cell[0] = float(line.split()[2])
            elif line.startswith("b ="):
                cell[1] = float(line.split()[2])
            elif line.startswith("c ="):
                cell[2] = float(line.split()[2])
            elif line.startswith("al ="):
                cell[3] = float(line.split()[2])
            elif line.startswith("be ="):
                cell[4] = float(line.split()[2])
            elif line.startswith("ga ="):
                cell[5] = float(line.split()[2])
        return np.array(cell, dtype='float32')

    @staticmethod
    def online_mean_variance(iterator):
        def update(count, mean, m2, value):
            count = count + 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2
            return count, mean, m2

        count, mean, m2 = 0, 0, 0
        for value in iterator:
            count, mean, m2 = update(count, mean, m2, value)

        variance = m2 / count
        return mean, variance

    def approximately_normalize(self, n=100):
        self.intensity_std = None
        self.metadata_mean = None
        self.metadata_std = None

        tfds = self.get_dataset()
        for (_,_,_,_,metadata,_,_),(intensity,) in tfds.ragged_batch(n):
            break

        self.intensity_std  = tf.math.reduce_std(intensity.flat_values, axis=0)

        self.metadata_mean = tf.reduce_mean(metadata.flat_values, axis=0)
        self.metadata_std  = tf.math.reduce_std(metadata.flat_values, axis=0)

    @staticmethod
    def get_average_cell(stream_file):
        def cell_iter(stream_file):
            for line in StreamLoader.stream_to_lines(stream_file):
                if line.startswith("Cell parameters"):
                    cell = line.split()
                    cell = np.array([
                        cell[2],
                        cell[3],
                        cell[4],
                        cell[6],
                        cell[7],
                        cell[8],
                    ], dtype='float32')
                    cell[:3] = 10.*cell[:3]
                    yield cell
                    break

        mean, variance = StreamLoader.online_mean_variance(cell_iter(stream_file))
        return mean

    @staticmethod
    def to_blocks(stream_file, block_name):
        """
        block_name : 'geometry', 'chunk', 'cell', 'peaks', 'crystal', or 'reflections'
        """
        block_begin_marker, block_end_marker = _block_markers[block_name]

        block = []
        in_block = False
        for line in StreamLoader.stream_to_lines(stream_file):
            if line.startswith(block_end_marker):
                in_block = False
                yield block
                block = []
            if in_block:
                block.append(line)
            if line.startswith(block_begin_marker):
                in_block = True

    @staticmethod
    def to_wavelengths_crystals(stream_file):
        """
        This method yields tuples (wavelength : float, crystal : list of strings) to handle XFEL data where the wavelength changes between shots.
        """
        block_begin_marker, block_end_marker = _block_markers['crystal']

        wavelength = None
        block = []
        in_block = False
        for line in StreamLoader.stream_to_lines(stream_file):
            if not in_block and line.startswith('photon_energy_eV'):
                eV = float(line.split()[-1])
                wavelength = rs.utils.ev2angstroms(eV)
            if line.startswith(block_end_marker):
                in_block = False
                yield wavelength, block
                block = []
            if in_block:
                block.append(line)
            if line.startswith(block_begin_marker):
                in_block = True

    @staticmethod
    def _crystal_to_data(crystal, cell, wavelength, dmin=None, intensity_std=None, metadata_mean=None, metadata_std=None, asu_id=0, spacegroup=None):
        inverse_wavelength = 1. / wavelength
        astar = bstar = cstar = None
        in_refls = False
        crystal_iter = iter(crystal)

        refls = []
        for line in crystal_iter:
            if line.startswith('astar ='):
                astar = (
                    np.array(line.split()[2:5], dtype="float32") / 10.0
                )  # crystfel's notation uses nm-1
            if line.startswith('bstar ='):
                bstar = (
                    np.array(line.split()[2:5], dtype="float32") / 10.0
                )  # crystfel's notation uses nm-1
            if line.startswith('cstar ='):
                cstar = (
                    np.array(line.split()[2:5], dtype="float32") / 10.0
                )  # crystfel's notation uses nm-1
            if line == "End of reflections\n":
                in_refls = False
            if in_refls:
                refls.append(line.split()[:-1])
            if line == "Reflections measured after indexing\n":
                in_refls = True
                crystal_iter = next(crystal_iter) #skip header

        refls = np.array(refls, dtype='float32')
        hkl = refls[:,:3]

        #Reflection resolution based on consensus cell
        d = cell.calculate_d_array(hkl).astype('float32')
        #Apply dmin
        if dmin is not None:
            idx = d >= dmin
            refls = refls[idx]
            d = d[idx]
            hkl = hkl[idx]
        if spacegroup is not None:
            idx = ~rs.utils.is_absent(hkl, spacegroup)
            refls = refls[idx]
            d = d[idx]
            hkl = hkl[idx]

        A = np.array([astar, bstar, cstar]).T
        # calculate ewald offset and s1

        s0 = np.array([0, 0, inverse_wavelength]).T
        q = hkl @ A.T # == (A @ hkl.T).T
        s1 = q + s0
        s1x, s1y, s1z = s1.T
        s1_norm = np.sqrt(s1x * s1x + s1y * s1y + s1z * s1z)

        # project calculated s1 onto the ewald sphere
        s1_obs = inverse_wavelength * s1 / s1_norm[:,None]

        # Compute the ewald offset vector
        eov = s1_obs - s1

        # Compute scalar ewald offset
        eo = s1_norm - inverse_wavelength

        # Compute angular ewald offset
        eo_sign = np.sign(eo)
        q_obs = s1_obs - s0
        ao = eo_sign * rs.utils.angle_between(q, q_obs)

        I = refls[:,3]
        SigI = refls[:,4]
        bg = refls[:,5]
        if intensity_std is not None:
            I = I / intensity_std
            SigI = SigI / intensity_std

        #panel = refls[:,9,None]
        inv_d2 = np.reciprocal(np.square(d))
        xydet = refls[:,7:9]
        #metadata = (s1[:,:2], eov, eo[...,None], ao[...,None], xydet, inv_d2[...,None])
        metadata = (s1[:,:2], eov)
        #metadata = (xydet,)
        metadata = np.concatenate(metadata, axis=-1).astype('float32')
        if metadata_mean is not None:
            metadata = metadata - metadata_mean
        if metadata_std is not None:
            metadata = metadata / metadata_std

        rowids = np.zeros_like(I, dtype='int32')
        d = tf.convert_to_tensor(d[None,:,None])
        I = tf.convert_to_tensor(I[None,:,None])
        SigI = tf.convert_to_tensor(SigI[None,:,None])
        hkl = tf.convert_to_tensor(hkl[None,:,:])
        metadata = tf.convert_to_tensor(metadata[None,:,:])

        asu = tf.ones_like(I, dtype='int32') * asu_id
        wavelength = tf.ones_like(I, dtype='float32') * wavelength

        data = ((asu, hkl, d, wavelength, metadata, I, SigI), (I,))
        return data

    def crystal_to_data(self, crystal, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        return StreamLoader._crystal_to_data(
            crystal, self.cell, wavelength, self.dmin, 
            self.intensity_std, self.metadata_mean, self.metadata_std,
            self.asu_id, self.spacegroup
        )

    def get_dataset(self):
        """
        Convert CrystFEL .stream files to a tf.data.Dataset.
        """
        def data_gen():
            for wavelength, lines in self.to_wavelengths_crystals(self.stream_file):
                #try:
                if self.wavelength is not None:
                    datum = self.crystal_to_data(lines, self.wavelength)
                else:
                    datum = self.crystal_to_data(lines, wavelength)
                #size = datum[0][0].values.shape[0]
                size = datum[0][0].shape[1]
                if size >= self.min_refls:
                    yield datum
                #except:
                #    print(f"Warning: encountered bad crystal block in {self.stream_file}")

        if self.signature is None:
            for datum in data_gen():
                self.signature = StreamLoader.get_signature_from_datum(datum)
                break

        return tf.data.Dataset.from_generator(data_gen, output_signature=self.signature).unbatch()

    @staticmethod
    def get_signature_from_datum(datum):
        """Work out the proper signature for creating a dataset from an example"""
        def to_ragged_spec(tensor):
            spec = tf.TensorSpec.from_tensor(tensor)
            shape,dtype = spec.shape,spec.dtype
            new = tf.TensorSpec(
                shape = (shape[0], None, shape[2]),
                dtype=dtype,
            )
            return new
        signature = tf.nest.map_structure(to_ragged_spec, datum)
        return signature

