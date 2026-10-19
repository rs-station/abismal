#/usr/bin/env cctbx.python
import numpy as np
import gemmi
from .loader import DataLoader, make_record, split_by_image



class StillsLoader(DataLoader):
    """ DIALS stills loader """
    def __init__(self, expt_files, refl_files, spacegroup=None, cell=None, dmin=None, asu_id=0, include_eo=True, cell_tol=None, isigi_cutoff=None):
        self.include_eo = include_eo
        if include_eo:
            super().__init__(6)
        else:
            super().__init__(3)
        # Activates a configured prefix if there is one, and raises with the
        # install instructions if there is not. Here as well as in the data
        # manager because the loader is usable on its own -- constructing it
        # directly otherwise fails on the deferred import below, several lines
        # from anything that explains why.
        from abismal.io._dials_check import require_dials

        require_dials(list(expt_files) + list(refl_files))

        from dxtbx.model.experiment_list import ExperimentListFactory  #defer defer defer defer
        self.expt_files = expt_files
        self.refl_files = refl_files
        self.dmin = dmin
        self.cell_tol = cell_tol
        self.isigi_cutoff = isigi_cutoff
        self.asu_id = asu_id

        elist_list = [ExperimentListFactory.from_json_file(expt_file, check_format=False) for expt_file in self.expt_files]
        self.cell,self.spacegroup = cell,spacegroup 
        if self.cell is None:
            self.cell = self.get_average_cell(elist_list)
        if self.spacegroup is None:
            self.spacegroup = self.get_space_group(elist_list)


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

    def get_dataset(self, num_cpus=1):
        """
        Lazily convert dials monochromatic stills files to per-image records.

        Returns a generator of 7-tuples of numpy arrays (see `DataLoader`),
        images in file order. With `num_cpus > 1` and several files, the files
        are parsed in a pool of processes (see `abismal.io.parallel`), and each
        file's images are yielded as soon as that file is parsed, without
        waiting for the files after it.
        """
        args = [
            (expt, refl, self.cell, self.spacegroup, self.dmin, self.asu_id,
             self.include_eo, self.cell_tol, self.isigi_cutoff)
            for expt,refl in zip(self.expt_files, self.refl_files)
        ]
        from abismal.io.parallel import ordered_map
        for flat in ordered_map(StillsLoader.dials_to_ragged, args, num_workers=num_cpus):
            yield from self._split_images(flat)

    @staticmethod
    def _split_images(flat):
        """Split one file's flat arrays, sorted by image, into image records."""
        image_ids, *columns = flat
        for record in split_by_image(image_ids, *columns):
            yield make_record(*record)

    @staticmethod
    def dials_to_ragged(expt_file, refl_file, cell, spacegroup, dmin, asu_id, include_eo=True, cell_tol=None, isigi_cutoff=None):
        """
        Convert dials monochromatic stills files to flat numpy arrays.

        Returns `(image_id, asu_id, hkl, d, wavelength, metadata, I, SigI)`,
        one row per reflection, sorted by `image_id`, which numbers the
        surviving images from zero. `StillsLoader._split_images` turns this
        into per-image records. Kept flat so that one array of each kind,
        rather than one per image, crosses between processes.
        """
        # Also here: dials_to_ragged is a staticmethod, and the pool runs it in
        # worker processes that never went through __init__.
        from abismal.io._dials_check import require_dials

        require_dials([expt_file, refl_file])

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
        s1 = np.array(Svec, dtype='float32')
        delpsical = np.array(table['delpsical.rad'])
        xy = s1 #actually xyz
        batch = table['id'].as_numpy_array()
        idx = ~spacegroup.operations().systematic_absences(np.array(h, dtype='int32'))
        bg = np.concatenate((
                np.array(table['background.sum.value'], dtype='float32')[:,None],
                np.sqrt(np.array(table['background.sum.variance'], dtype='float32'))[:,None],
            ), axis=-1,
        )

        I  = np.array(table['intensity.sum.value'], dtype='float32')
        SigI  = np.array(np.sqrt(table['intensity.sum.variance']), dtype='float32')

        if dmin is not None:
            idx &= d >= dmin

        if isigi_cutoff is not None:
            isigi = I / SigI
            idx &= isigi >= isigi_cutoff

        if cell_tol is not None:
            ref_cell = np.array(cell.parameters)
            cells = np.array([C.get_unit_cell().parameters() for C in elist.crystals()])
            frac_diff = np.abs(cells - ref_cell) / ref_cell
            idx &= np.all(frac_diff <= cell_tol, axis=1)[batch]

        hkl = hkl[idx]
        d = d[idx, None]
        wavelength = wavelength[idx, None]
        dQ = dQ[idx]
        xy = xy[idx]
        delpsical = delpsical[idx, None]
        batch = batch[idx]
        I = I[idx, None]
        SigI = SigI[idx, None]
        bg = bg[idx]

        #metadata = [xy, bg, hkl]
        metadata = [xy, bg]
        if include_eo:
            metadata.extend([
                delpsical, dQ,
            ])
        metadata = np.concatenate(metadata, axis=-1)

        # Images are numbered in table order and the table is grouped by
        # experiment id, but make sure: _split_images needs sorted ids. The
        # stable sort keeps each image's reflections in table order.
        batch = np.unique(batch, return_inverse=True)[1].astype(batch.dtype)
        order = np.argsort(batch, kind='stable')
        asu = np.full(len(batch), asu_id, dtype='int32')
        data = (batch, asu, hkl, d, wavelength, metadata, I, SigI)
        data = tuple(x[order] for x in data)

        return data

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
