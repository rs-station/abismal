#/usr/bin/env cctbx.python
import mmap
import numpy as np
import gemmi
from abismal.io.crystfel import StreamLoaderBase, _parse_chunk_group
from abismal.io.parallel import ordered_map
from abismal.io.loader import make_record
from reciprocalspaceship.decorators import spacegroupify,cellify




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

    def _convert_to_numpy(self, peak_list, wavelength):
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

        return make_record(self.asu_id, hkl, d, wavelength, metadata, I, SigI)

    def _parse_chunk(self, *args, **kwargs):
        data = super()._parse_chunk(*args, **kwargs)
        result = [self._convert_to_numpy(pl, data['wavelength']) for pl in data['peak_lists'] if len(pl) > 0]
        result = [pl for pl in result if pl is not None]
        return result

    def get_dataset(self, peak_list_columns=None, **ray_kwargs):
        """
        Lazily parse the CrystFEL .stream file into per-image records.

        Returns a generator of 7-tuples of numpy arrays (see `DataLoader`).
        Nothing is read until it is iterated; with `num_cpus > 1` the first
        `next` starts a pool of parsing processes (see `abismal.io.parallel`).
        """

        if peak_list_columns is None:
            peak_list_columns = [
                'H', 'K', 'L', 
                'I', 'SigI', 
                'H', 'K', 'L', 
                'background', 
                's1x', 's1y', 's1z', 
                'ewald_offset', 'angular_ewald_offset',
                'ewald_offset_x', 'ewald_offset_y', 'ewald_offset_z',
            ]

        chunks = self.read_crystfel(
            wavelength=self.wavelength,
            peak_list_columns=peak_list_columns,
            **ray_kwargs,
        )

        # Chunks come back in file order, each as soon as it is parsed, so a
        # consumer can start on the first images while the rest are parsed.
        for chunk in chunks:
            for peak_list in chunk:
                yield peak_list

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
        Parse a CrystFEL stream file, chunks in groups, in a pool of processes when
        `num_cpus > 1` (see `abismal.io.parallel`), yielding chunks in file order
        as they are parsed. Other keyword arguments are accepted and ignored.

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
        num_cpus : int (optional)
            The number of parsing processes; one (serial) by default.
        use_ray, address, ray_kwargs : optional
            Accepted for compatibility and ignored.

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

        workers = num_cpus if num_cpus is not None and num_cpus > 1 else 1
        tasks = [(self, spans, wavelength, chunk_metadata_keys, crystal_metadata_keys, peak_list_columns)
                 for spans in self._chunk_groups(workers)]
        for group in ordered_map(_parse_chunk_group, tasks, num_workers=workers):
            yield from group
