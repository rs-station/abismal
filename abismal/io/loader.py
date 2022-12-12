#!/usr/bin/env cctbx.python
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf


class DataLoader():
    data_labels = (
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


        ds.compute_dHKL(True)
        ds['rowids'] = ds.groupby(self.batch_key).ngroup().to_numpy("int32")
        ds.sort_values("rowids", inplace=True)
        rowids = ds.rowids.to_numpy("int32")

        I = tf.RaggedTensor.from_value_rowids(ds[self.intensity_key].to_numpy("float32")[:,None], rowids)
        d = tf.RaggedTensor.from_value_rowids(ds['dHKL'].to_numpy("float32")[:,None], rowids)
        SigI = tf.RaggedTensor.from_value_rowids(ds[self.sigma_key].to_numpy("float32")[:,None], rowids)
        Metadata = tf.RaggedTensor.from_value_rowids(ds[self.metadata_keys].to_numpy("float32"), rowids)
        HKL = tf.RaggedTensor.from_value_rowids(ds.get_hkls(), rowids)
        wavelength = tf.ones_like(I) * self.wavelength

        tfds = tf.data.Dataset.from_tensor_slices(
            ((HKL, d, wavelength, Metadata, I, SigI), (I,)),
        )

        return tfds

    def data_gen(self):
        ds = self.ds[self.ds.dHKL >= self.dmin]
        batch_key = self.batch_key
        intensity_key = self.intensity_key
        sigma_key = self.sigma_key
        metadata_keys = self.metadata_keys
        max_reflections_per_image = self.max_reflections_per_image
        min_reflections_per_image = self.min_reflections_per_image

        max_reflections_per_image = ds.groupby("BATCH").size().max()
        ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()

        #Standardize the metadata
        ds[metadata_keys] = (ds[metadata_keys] - ds[metadata_keys].mean()) / ds[metadata_keys].std()

        for batch,im in ds.groupby(batch_key):
            if len(im) > max_reflections_per_image:
                im = im.iloc[:max_reflections_per_image]
            metadata = im[metadata_keys].to_numpy('float32') 
            iobs = im[intensity_key].to_numpy('float32')[:,None]
            sigiobs = im[sigma_key].to_numpy('float32')[:,None]
            hkl = im.get_hkls().astype('int32')

            #Pad to the same length
            n = max_reflections_per_image - len(im)
            mask = tf.ones((len(im), 1), dtype='float32')
            mask = tf.pad(mask, [[0, n], [0, 0]])
            hkl = tf.pad(hkl, [[0, n], [0, 0]])
            metadata = tf.pad(metadata, [[0, n], [0, 0]])
            iobs = tf.pad(iobs, [[0, n], [0, 0]])
            sigiobs = tf.pad(sigiobs, [[0, n], [0, 0]], constant_values=1.)
            mask = mask

            inputs  = (hkl, metadata, iobs, sigiobs, mask)
            targets = (iobs,)
            yield inputs, targets

class StillsLoader(DataLoader):
    """ DIALS stills loader """
    def __init__(self, expt_files, refl_files, spacegroup=None, cell=None, dmin=None):
        super().__init__(5)
        from dxtbx.model.experiment_list import ExperimentListFactory  #defer defer defer defer
        self.expt_files = expt_files
        self.refl_files = refl_files
        self.dmin = dmin

        # Bwaaaaahhh listlist
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
        metadata = np.concatenate((xy, dQ), axis=-1)

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

        data = ((hkl, d, wavelength, metadata, I, SigI), (I,))

        return data

