#/usr/bin/env cctbx.python
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf
from .loader import DataLoader


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
