#/usr/bin/env cctbx.python
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf
from abismal.io.loader import DataLoader
from reciprocalspaceship.decorators import spacegroupify,cellify
from multiprocessing import cpu_count,Pool
from abismal.io.crystfel import StreamLoader 
from abismal.io.crystfel import read_crystfel




class StreamDataLoader(DataLoader):
    @cellify
    @spacegroupify
    def __init__(self, stream_file, cell=None, dmin=None, asu_id=0, wavelength=None, encoding='utf-8', isigi_cutoff=None):
        """
        Parameters
        ----------
        stream_file : str
        cell : gemmi.UnitCell (optional)
            It is important that a consistent cell is used for resolution calculations. 
            If a cell is supplied it will be used. Otherwise, this class will first try to 
            located a unit cell record in the stream file header. Failing that, the average
            cell across the data set will be calculated. If a lattice type is availabe, the
            symmetry of the lattice will be used to correct cell axis lengths and angles. 
        dmin : float (optional)
        asu_id : int (optional)
            asu_id will be passed to the downstream model and is used to control which reflections 
            are merged together. 
        wavelength : float (optional)
            Optionally override the wavelengths recorded for each event in the stream file. 
        encoding : str (optional)
            Try to override the default encoding, 'utf-8'
        isigi_cutoff : float (optional)
            Discard reflections with I/Sigma less than this threshold.
        min_refls : int (optional)
            Discard images with fewer reflections
        """
        self.peak_list_columns = [
            'H', 'K', 'L', 
            'I', 'SigI', 
            's1x', 's1y', 's1z', 
            'ewald_offset_x', 'ewald_offset_y', 'ewald_offset_z',
        ]
        super().__init__(len(self.peak_list_columns) - 5)
        self.stream_file = stream_file
        self.cell = self._get_unit_cell(cell)
        self.asu_id = asu_id
        self.wavelength = wavelength
        self.dmin = dmin
        self.isigi_cutoff = isigi_cutoff

    def _get_unit_cell(self, cell=None):
        sl = StreamLoader(self.stream_file)
        if cell is None:
            cell = sl.extract_target_unit_cell()
            
        if cell is None:
            cell = sl.calculate_average_unit_cell()
        if not isinstance(cell, gemmi.UnitCell):
            cell = gemmi.UnitCell(*cell)
        return cell

    def get_dataset(self, num_cpus=1, **kwargs):
        """
        Convert CrystFEL .stream files to a tf.data.Dataset.
        """
        ds = read_crystfel(self.stream_file, num_cpus=num_cpus, columns=self.peak_list_columns)
        ds.cell = self.cell
        ds.compute_dHKL(inplace=True)
        if self.dmin is not None:
            ds.drop(ds.dHKL < self.dmin, inplace=True)
        if self.isigi_cutoff is not None:
            ds.drop(ds.I / ds.SigI <= self.isigi_cutoff, inplace=True)
        ds['ASU'] = self.asu_id
        _,ds['BATCH'] = np.unique(ds.BATCH, return_inverse=True)

        batch = ds['BATCH'].to_numpy('int64')
        ds.reset_index(inplace=True)
        asu = tf.RaggedTensor.from_value_rowids(
            ds['ASU'].to_numpy('int64')[:,None],
            batch,
        )
        hkl = tf.RaggedTensor.from_value_rowids(
            ds[['H', 'K', 'L']].to_numpy('int64'),
            batch,
        )
        d = tf.RaggedTensor.from_value_rowids(
            ds['dHKL'].to_numpy('float32')[:,None],
            batch,
        )
        wavelength = tf.RaggedTensor.from_value_rowids(
            ds['Wavelength'].to_numpy('float32')[:,None],
            batch,
        )
        I = tf.RaggedTensor.from_value_rowids(
            ds['I'].to_numpy('float32')[:,None],
            batch,
        )
        SigI = tf.RaggedTensor.from_value_rowids(
            ds['SigI'].to_numpy('float32')[:,None],
            batch,
        )
        metadata = tf.RaggedTensor.from_value_rowids(
            ds[self.peak_list_columns[5:]].to_numpy('float32'),
            batch,
        )

        data = ((asu, hkl, d, wavelength, metadata, I, SigI), (I,))
        dataset = tf.data.Dataset.from_tensors(data)
        return dataset.unbatch()


if __name__ == '__main__':
    file = "/mnt/raid/data/xtal/abismal_examples/cxidb_62/all-amb.stream.bz2"

    dmin = 1.5
    sg = "P 65"
    loader = StreamLoader(file, dmin=dmin, spacegroup=sg)
    ds = loader.get_dataset()
    count_max = 100
    count = 0
    for datum in ds:
        count += 1
        if count > count_max:
            break

    from IPython import embed
    embed(colors='linux')
