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




class StreamLoader(rs.io.crystfel.StreamLoader):
    @cellify
    @spacegroupify
    def __init__(self, stream_file, cell=None, dmin=None, asu_id=0, wavelength=None, encoding='utf-8'):
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
        """
        super().__init__(stream_file, encoding)

        self.signature = None
        self.cell = self._get_unit_cell(cell)
        self.asu_id = asu_id
        self.wavelength = wavelength
        self.dmin = dmin

    def _get_unit_cell(self, cell=None):
        if cell is None:
            cell = self.extract_target_unit_cell()
        if cell is None:
            cell = self.calculate_average_unit_cell()
        return cell

    def _convert_to_tf(self, peak_list, wavelength):
        hkl = peak_list[:,:3]
        d = self.cell.calculate_d_array(hkl)
        I = peak_list[:,3]
        SigI = peak_list[:,4]
        metadata = peak_list[:,5:]

        if self.dmin is not None:
            idx = d >= self.dmin
            d = d[idx]
            hkl = hkl[idx]
            peak_list = peak_list[idx]
            I = I[idx]
            SigI = SigI[idx]
            metadata = metadata[idx]

        I = tf.convert_to_tensor(I[None,:,None])
        d = tf.convert_to_tensor(d[None,:,None])
        SigI = tf.convert_to_tensor(SigI[None,:,None])
        hkl = tf.convert_to_tensor(hkl[None,:,:], dtype='int32')
        metadata = tf.convert_to_tensor(metadata[None,:,:])
        asu = tf.ones_like(I, dtype='int32') * self.asu_id
        wavelength = tf.ones_like(I) * wavelength

        data = ((asu, hkl, d, wavelength, metadata, I, SigI), (I,))
        return data

    def _parse_chunk(self, *args, **kwargs):
        data = super()._parse_chunk(*args, **kwargs)
        return [self._convert_to_tf(pl, data['wavelength']) for pl in data['peak_lists']]

    def get_dataset(self, peak_list_columns=None, **ray_kwargs):
        """
        Convert CrystFEL .stream files to a tf.data.Dataset.
        """

        if peak_list_columns is None:
            peak_list_columns = [
                'H', 'K', 'L', 
                'I', 'SigI', 
                's1x', 's1y', 's1z', 
                'ewald_offset_x', 'ewald_offset_y', 'ewald_offset_z',
            ]

        chunks = self.parallel_read_crystfel(
            wavelength=self.wavelength,
            peak_list_columns=peak_list_columns,
            **ray_kwargs,
        )

        def data_gen():
            for chunk in chunks:
                for peak_list in chunk:
                    yield peak_list

        if self.signature is None:
            for datum in data_gen():
                self.signature = StreamLoader.get_signature_from_datum(datum)
                break

        return tf.data.Dataset.from_generator(data_gen, output_signature=self.signature).unbatch()

    @staticmethod
    def get_signature_from_datum(datum):
        """Work out the proper signature for creating a dataset from an example"""
        #if ragged:
        #    signature = tf.nest.map_structure(tf.RaggedTensorSpec.from_value, datum)
        #else:
        #    signature = tf.nest.map_structure(tf.TensorSpec.from_tensor, datum)
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
