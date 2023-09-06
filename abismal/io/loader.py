#/usr/bin/env cctbx.python
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf


class DataLoader():
    data_labels = (
        "ASU",
        "HKL",
        "Resolution",
        "Wavelength",
        "Metadata",
        "I",
        "SigI",
    ) #To support laue add wavelength and resolution here
    def __init__(self, metadata_length):
        self.signature=(
            (
                tf.RaggedTensorSpec((None, None, 1), tf.int32, 1, tf.int32), #ASU ID
                tf.RaggedTensorSpec((None, None, 3), tf.int32, 1, tf.int32), #HKL 
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #dHKL
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #wavelength
                tf.RaggedTensorSpec((None, None, metadata_length), tf.float32, 1, tf.int32), #metadata
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #I
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #SigI
            ),(
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #I 
            )
        )

def get_first_key_of_type(ds, dtype):
    idx = ds.dtypes == dtype
    if idx.sum() == 0:
        raise ValueError(f"Dataset has no key of type {dtype}")
    key = ds.dtypes[idx].keys()[0]
    return key

class MTZLoader(DataLoader):
    def __init__(
            self, 
            mtz_file, 
            wavelength=1.,
            dmin=None,
            metadata_keys=None, 
            cell=None, 
            spacegroup=None, 
            batch_key=None, 
            intensity_key=None,
            sigma_key=None,
            asu_id=0,
        ):
        self.metadata_keys = metadata_keys
        if self.metadata_keys is None:
            self.metadata_keys = [
                "XDET",
                "YDET",
            ]

        super().__init__(len(self.metadata_keys))

        self.wavelength = wavelength
        self.mtz_file = mtz_file
        self.dmin = dmin
        self.cell = cell
        self.spacegroup = spacegroup
        self.asu_id = asu_id

        self.batch_key = batch_key
        self.intensity_key = intensity_key
        self.sigma_key = sigma_key

    def get_dataset(self):
        ds = rs.read_mtz(self.mtz_file)

        if self.cell is None:
            self.cell = ds.cell
        else:
            ds.cell = self.cell

        if self.spacegroup is None:
            self.spacegroup = ds.spacegroup
        else:
            ds.spacegroup = self.spacegroup

        if self.batch_key is None:
            self.batch_key = get_first_key_of_type(ds, 'B')
        if self.intensity_key is None:
            self.intensity_key = get_first_key_of_type(ds, 'J')
        if self.sigma_key is None:
            self.sigma_key = get_first_key_of_type(ds, 'Q')


        ds.compute_dHKL(True).label_absences(True)
        ds = ds[~ds.ABSENT]
        if self.dmin is not None:
            ds = ds[ds.dHKL >= self.dmin]
        ds['rowids'] = ds.groupby(self.batch_key).ngroup().to_numpy("int32")
        ds.sort_values("rowids", inplace=True)
        rowids = ds.rowids.to_numpy("int32")

        I = tf.RaggedTensor.from_value_rowids(ds[self.intensity_key].to_numpy("float32")[:,None], rowids)
        d = tf.RaggedTensor.from_value_rowids(ds['dHKL'].to_numpy("float32")[:,None], rowids)
        SigI = tf.RaggedTensor.from_value_rowids(ds[self.sigma_key].to_numpy("float32")[:,None], rowids)
        Metadata = tf.RaggedTensor.from_value_rowids(ds[self.metadata_keys].to_numpy("float32"), rowids)
        HKL = tf.RaggedTensor.from_value_rowids(ds.get_hkls(), rowids)
        wavelength = tf.ones_like(I) * self.wavelength
        ASU = tf.ones_like(I, dtype='int32') * self.asu_id

        tfds = tf.data.Dataset.from_tensor_slices(
            ((ASU, HKL, d, wavelength, Metadata, I, SigI), (I,)),
        )

        return tfds

class StillsLoader(DataLoader):
    """ DIALS stills loader """
    def __init__(self, expt_files, refl_files, spacegroup=None, cell=None, dmin=None, asu_id=0, include_eo=True):
        self.include_eo = include_eo
        if include_eo:
            super().__init__(5)
        else:
            super().__init__(2)
        from dxtbx.model.experiment_list import ExperimentListFactory  #defer defer defer defer
        self.expt_files = expt_files
        self.refl_files = refl_files
        self.dmin = dmin
        self.asu_id = asu_id

        elist_list = [ExperimentListFactory.from_json_file(expt_file, check_format=False) for expt_file in self.expt_files]
        self.cell,self.spacegroup = cell,spacegroup 
        if self.cell is None:
            self.cell = self.get_average_cell(elist_list)
        if self.spacegroup is None:
            self.spacegroup = self.get_space_group(elist_list)

        self.mean = None
        self.std  = None

    @staticmethod
    def get_average_cell(elist_list):
        """
        Parameters
        ----------
        elist_list : list
            A list of DIALS experiment lists 

        Returns
        -------
            cell : gemmi.UnitCell
        """
        cell = np.zeros(6)
        l = 0
        from tqdm import tqdm
        print("Determining unit cell ...")
        for elist in tqdm(elist_list):
            crystals = elist.crystals()
            cell += np.array([c.get_unit_cell().parameters() for c in crystals]).sum(0)
            l += len(crystals)
        cell = cell/l
        cell = gemmi.UnitCell(*cell)
        print(f"Average cell: {cell}")
        return cell

    @staticmethod
    def get_space_group(elist_list, check_consistent=False):
        """
        This method will raise a `ValueError` if all the space groups do not match. 

        Parameters
        ----------
        elist_list : list
            A list of DIALS experiment lists 

        Returns
        -------
            spacegroup :  gemmi.SpaceGroup
        """
        hms = elist_list[0].crystals()[0].get_space_group().type().universal_hermann_mauguin_symbol()
        if not check_consistent:
            return gemmi.SpaceGroup(hms)

        from tqdm import tqdm
        for elist in tqdm(elist_list):
            crystals = elist.crystals()
            _hms = np.array([c.get_space_group().type().universal_hermann_mauguin_symbol() for c in crystals])
            if not np.all(_hms == hms):
                cid = np.where(_hms != hms)[0][0]
                raise ValueError(f"Crystal {cid} has Universal Hermann Mauguin symbol {_hms[cid]} but {hms} was expected")
        return gemmi.SpaceGroup(hms)

    def _load_expt_refl(self, expt_and_refl):
        expt = expt_and_refl[0]
        refl = expt_and_refl[1]
        ragged = self.dials_to_ragged(expt, refl)
        ds = tf.data.Dataset.from_tensor_slices(ragged)
        return ds

    def get_dataset(self):
        """
        Convert dials monochromatic stills files to a tf.data.Dataset.
        """
        def data_gen():
            for expt,refl in list(zip(self.expt_files, self.refl_files)):
                data = self.dials_to_ragged(expt, refl)
                yield data
        return tf.data.Dataset.from_generator(data_gen, output_signature=self.signature).unbatch()

    def dials_to_ragged(self, expt_file, refl_file):
        """
        Convert dials monochromatic stills files to ragged tensors.
        """
        from dials.array_family import flex
        from dxtbx.model.experiment_list import ExperimentListFactory 
        table = flex.reflection_table().from_file(refl_file)
        elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)

        idx = flex.size_t(np.array(table['id']))

        table.compute_d(elist)
        table["A_matrix"] = flex.mat3_double( [C.get_A() for C in elist.crystals()] ).select(idx)
        table["s0_vec"] = flex.vec3_double( [e.beam.get_s0() for e in elist] ).select(idx)
        table["wavelength"] = flex.double( [e.beam.get_wavelength() for e in elist] ).select(idx)

        h = table["miller_index"].as_vec3_double()
        Q = table["A_matrix"] * h
        Svec = Q + table["s0_vec"]
        Qobs = table['s0_vec'].norms() * Svec / Svec.norms() - table['s0_vec']

        hkl = np.array(h, dtype='int32')
        d = np.array(table['d'], dtype='float32')
        wavelength = np.array(table['wavelength'], dtype='float32')
        dQ = np.array(Q - Qobs, dtype='float32')
        xy = np.array(Svec, dtype='float32')[:,:2]
        batch = table['id'].as_numpy_array()
        idx = ~self.spacegroup.operations().systematic_absences(h)
        if self.dmin is not None:
            idx &= d >= self.dmin

        I  = np.array(table['intensity.sum.value'], dtype='float32')
        SigI  = np.array(np.sqrt(table['intensity.sum.variance']), dtype='float32')

        hkl = hkl[idx]
        d = d[idx, None]
        wavelength = wavelength[idx, None]
        dQ = dQ[idx]
        xy = xy[idx]
        batch = batch[idx]
        I = I[idx, None]
        SigI = SigI[idx, None]
        if self.include_eo:
            metadata = np.concatenate((xy, dQ), axis=-1)
        else:
            metadata = xy

        if self.mean is None:
            self.mean = (metadata.mean(0), I.mean(0), SigI.mean(0))
            self.std  = (metadata.std(0), I.std(0), SigI.std(0))

        metadata = (metadata - self.mean[0]) / self.std[0]
        I = I / self.std[1]
        SigI = SigI / self.std[1]

        hkl = tf.RaggedTensor.from_value_rowids(hkl, batch)
        d = tf.RaggedTensor.from_value_rowids(d, batch)
        wavelength = tf.RaggedTensor.from_value_rowids(wavelength, batch)
        metadata = tf.RaggedTensor.from_value_rowids(
            metadata,
            batch,
        )
        I = tf.RaggedTensor.from_value_rowids(I, batch)
        SigI = tf.RaggedTensor.from_value_rowids(SigI, batch)
        asu = tf.ones_like(I, dtype='int32') * self.asu_id

        data = ((asu, hkl, d, wavelength, metadata, I, SigI), (I,))

        return data

class StreamLoader(DataLoader):
    def __init__(self, stream_file, spacegroup=None, cell=None, dmin=None, asu_id=0):
        super().__init__(5)

        if cell is None:
            self.get_average_cell(stream_file)

    @staticmethod
    def to_crystals(stream_file):
        return StreamLoader.to_blocks(stream_file, 'crystal')


    @staticmethod
    def get_cell(stream_file):
        geo = next(StreamLoader.to_blocks(stream_file, 'geometry'))
        for line in geo:
            if line.startswith("photon_energy"):
                eV = float(line.split()[2])
                lam = rs.utils.ev2angstroms(eV)
                return lam

    @staticmethod
    def get_wavelength(stream_file):
        geo = next(StreamLoader.to_blocks(stream_file, 'geometry'))
        for line in geo:
            if line.startswith("photon_energy"):
                eV = float(line.split()[2])
                lam = rs.utils.ev2angstroms(eV)
                return lam

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

    @staticmethod
    def get_average_cell(stream_file):
        def cell_iter(stream_file):
            for crystal in StreamLoader.to_crystals(stream_file):
                for line in crystal:
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
    def get_average_cell(stream_file):
        def cell_iter(stream_file):
            for crystal in StreamLoader.to_crystals(stream_file):
                for line in crystal:
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
        # See crystFEL API reference here: https://www.desy.de/~twhite/crystfel/reference/stream_8h.html
        block_markers = {
            "geometry" : ("----- Begin geometry file -----", "----- End geometry file -----"),
            "chunk" : ("----- Begin chunk -----", "----- End chunk -----"),
            "cell" : ("----- Begin unit cell -----", "----- End unit cell -----"),
            "peaks" : ("Peaks from peak search", "End of peak list"),
            "crystal" : ("--- Begin crystal", "--- End crystal"),
            "reflections" : ("Reflections measured after indexing", "End of reflections"),
        }
        block_begin_marker, block_end_marker = block_markers[block_name]

        block = []
        in_block = False
        for line in open(stream_file):
            if line.startswith(block_end_marker):
                in_block = False
                yield block
                block = []
            if in_block:
                block.append(line)
            if line.startswith(block_begin_marker):
                in_block = True

if __name__=='__main__':
    stream_file = '/mnt/raid/data/xtal/20210615_Neutze_collab/indexing_dark_before_merging_intensities.stream'
    test = np.random.random((1_000, 6))
    mean, variance = StreamLoader.online_mean_variance(test)
    assert np.all(np.isclose(mean, test.mean(0)))
    assert np.all(np.isclose(variance, np.square(test.std(0))))

    for crystal in StreamLoader.to_blocks(stream_file, 'crystal'):
        break

    assert len(crystal) > 0

    cell = StreamLoader.get_average_cell(stream_file)

    geo = StreamLoader.to_blocks(stream_file, 'geometry')
    print(''.join(next(geo)))

    print(StreamLoader.get_wavelength(stream_file))
