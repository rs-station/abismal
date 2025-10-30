#/usr/bin/env cctbx.python
import mmap
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf
from abismal.io.common import check_for_ray,ray_context
from abismal.io.loader import DataLoader
from abismal.io.crystfel import StreamLoaderBase
from reciprocalspaceship.decorators import spacegroupify,cellify
from multiprocessing import cpu_count,Pool




class StreamLoader(StreamLoaderBase):
    @cellify
    @spacegroupify
    def __init__(self, stream_file, cell=None, dmin=None, asu_id=0, wavelength=None, encoding='utf-8', isigi_cutoff=None, cell_tol=None):
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
        """
        super().__init__(stream_file, encoding)
        self.cell_tol = cell_tol
        self.signature = None
        self.cell = self._get_unit_cell(cell)
        self.asu_id = asu_id
        self.wavelength = wavelength
        self.dmin = dmin
        self.isigi_cutoff = isigi_cutoff

    def filter_peak_list(self, data):
        if self.cell_tol is None:
            return False
        cell = self.re_crystal_metadata['Cell parameters'].findall(data)[0].split()
        cell = np.array([float(cell[i]) for i in (0, 1, 2, 4, 5, 6)])
        cell[:3] *= 10.
        target = self.cell.parameters
        r = np.abs(cell - target) / target
        if (r <= self.cell_tol).all():
            return False
        return True

    def _get_unit_cell(self, cell=None):
        if cell is None:
            cell = self.extract_target_unit_cell()
            
        if cell is None:
            cell = self.calculate_average_unit_cell()
        if not isinstance(cell, gemmi.UnitCell):
            cell = gemmi.UnitCell(*cell)
        return cell

    def _convert_to_tf(self, peak_list, wavelength):
        hkl = peak_list[:,:3]
        d = self.cell.calculate_d_array(hkl)
        I = peak_list[:,3]
        SigI = peak_list[:,4]
        metadata = peak_list[:,5:]

        if self.dmin is not None or self.isigi_cutoff is not None:
            idx = np.ones_like(d, dtype=bool)
            if self.dmin is not None:
                idx &= d >= self.dmin
            if self.isigi_cutoff is not None:
                isigi = I / SigI
                idx &= isigi >= self.isigi_cutoff

            if not idx.any():
                return None

            d = d[idx]
            hkl = hkl[idx]
            peak_list = peak_list[idx]
            I = I[idx]
            SigI = SigI[idx]
            metadata = metadata[idx]

        I = tf.convert_to_tensor(I[None,:,None])
        d = tf.convert_to_tensor(d[None,:,None])
        SigI = tf.convert_to_tensor(SigI[None,:,None])
        hkl = tf.convert_to_tensor(hkl[None,:,:], dtype='int64')
        metadata = tf.convert_to_tensor(metadata[None,:,:])
        asu = tf.ones_like(I, dtype='int64') * self.asu_id
        wavelength = tf.ones_like(I) * wavelength

        data = ((asu, hkl, d, wavelength, metadata, I, SigI), (I,))
        return data

    def _parse_chunk(self, *args, **kwargs):
        data = super()._parse_chunk(*args, **kwargs)
        result = [self._convert_to_tf(pl, data['wavelength']) for pl in data['peak_lists'] if len(pl) > 0]
        result = [pl for pl in result if pl is not None]
        return result

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

        chunks = self.read_crystfel(
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

    def read_crystfel(
        self,
        wavelength=None, chunk_metadata_keys=None,
        crystal_metadata_keys=None,
        peak_list_columns=None,
        use_ray=True,
        num_cpus=None,
        address="local",
        **ray_kwargs,
    ) -> list:
        """
        Parse a CrystFEL stream file using multiple processors. Parallelization depends on the ray library (https://www.ray.io/).
        If ray is unavailable, this method falls back to serial processing on one CPU. Ray is not a dependency of reciprocalspaceship
        and will not be installed automatically. Users must manually install it prior to calling this method.

        PARAMETERS
        ----------
        wavelength : float
            Override the wavelength with this value. Wavelength is used to compute Ewald offsets.
        chunk_metadata_keys : list
            A list of metadata_keys which will be returned in the resulting dictionaries under the 'chunk_metadata' entry.
            A list of possible keys is stored as stream_loader.available_chunk_metadata_keys
        crytal_metadata_keys : list
            A list of metadata_keys which will be returned in the resulting dictionaries under the 'crystal_metadata' entry.
            A list of possible keys is stored as stream_loader.available_crystal_metadata_keys
        peak_list_columns : list
            A list of columns to include in the peak list numpy arrays.
            A list of possible column names is stored as stream_loader.available_column_names.
        use_ray : bool(optional)
            Whether or not to use ray for parallelization.
        num_cpus : int (optional)
            The number of cpus for ray to use.
        ray_kwargs : optional
            Additional keyword arguments to pass to [ray.init](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray.init).

        RETURNS
        -------
        chunks : list
            A list of dictionaries containing the per-chunk data. The 'peak_lists' item contains a
            numpy array with shape n x 14 with the following information.
                h, k, l, I, SIGI, peak, background, fs/px, ss/px, s1x, s1y, s1z,
                ewald_offset, angular_ewald_offset
        """
        if peak_list_columns is not None:
            peak_list_columns = [self.peak_list_columns[s] for s in peak_list_columns]

        # Check whether ray is available
        use_ray = False
        if num_cpus > 1:
            use_ray = check_for_ray()

        with open(self.filename, "r") as f:
            memfile = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            beginnings_and_ends = zip(
                self.block_regex_bytes["chunk_begin"].finditer(memfile),
                self.block_regex_bytes["chunk_end"].finditer(memfile),
            )
            if use_ray:
                with ray_context(num_cpus=num_cpus, **ray_kwargs) as ray:

                    @ray.remote
                    def parse_chunk(loader: StreamLoaderBase, *args):
                        return loader._parse_chunk(*args)

                    result_ids = []
                    for begin, end in beginnings_and_ends:
                        result_ids.append(
                            parse_chunk.remote(
                                self,
                                begin.start(),
                                end.end(),
                                wavelength,
                                chunk_metadata_keys,
                                crystal_metadata_keys,
                                peak_list_columns,
                            )
                        )

                    for result_id in result_ids:
                        yield ray.get(result_id)

            else:
                for begin, end in beginnings_and_ends:
                    yield self._parse_chunk(
                        begin.start(),
                        end.end(),
                        wavelength,
                        chunk_metadata_keys,
                        crystal_metadata_keys,
                        peak_list_columns,
                    )

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

