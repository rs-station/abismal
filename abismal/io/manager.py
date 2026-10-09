from itertools import chain
from reciprocalspaceship.decorators import spacegroupify,cellify
from abismal.io.pipeline import ImageCache, BatchStream
import numpy as np
import reciprocalspaceship as rs
import yaml

# TODO: refactor this filetype control flow into abismal.io
_file_endings = {
    'refl' : ('.refl', '.pickle'),
    'expt' : ('.expt', '.json'),
    'stream' : ('.stream',),
    'mtz' : ('.mtz',),
}

def _is_file_type(s, endings):
    for ending in endings:
        if s.endswith(ending):
            return True
    return False

def _is_stream_file(s):
    return _is_file_type(s, _file_endings['stream'])

def _is_refl_file(s):
    return _is_file_type(s, _file_endings['refl'])

def _is_expt_file(s):
    return _is_file_type(s, _file_endings['expt'])

def _is_mtz_file(s):
    return _is_file_type(s, _file_endings['mtz'])

def _is_dials_file(s):
    return _is_refl_file(s) or _is_expt_file(s)

def _split_metadata_keys(value):
    """Turn the CLI's comma-separated string into the list MTZLoader wants.

    Parsed here rather than in DataManager so the manager's own API takes a
    list, which is what `MTZLoader` and the yaml config both use.
    """
    if value is None:
        return None
    keys = [k.strip() for k in value.split(",") if k.strip()]
    return keys or None


class DataManager:
    """
    A high-level class for managing I/O of reflection data.
    """
    def __init__(self, inputs, dmin, cell=None, spacegroup=None, 
            num_cpus=None, separate=False, wavelength=None, ray_log_level="ERROR",
            test_fraction=0., separate_friedel_mates=False, cell_tol=None, isigi_cutoff=None, 
            shuffle_buffer_size=0, batch_size=100, steps_per_epoch=None, validation_steps=None,
            epochs=30, ambiguate=False, mtz_metadata=None, reflections_per_image=None,
            sample_with_replacement=True, seed=None,
        ):
        if separate_friedel_mates and separate:
            raise ValueError("Cannot combine --separate-friedel-mates and --separate")

        self.inputs = inputs
        self.dmin = dmin
        self.wavelength = wavelength
        self.separate = separate
        self.num_cpus = num_cpus
        self.ray_log_level = ray_log_level
        self.cell = cell
        self.spacegroup = spacegroup
        self.test_fraction = test_fraction
        self.num_asus = 0
        self.separate_friedel_mates = separate_friedel_mates
        self.cell_tol = cell_tol
        self.isigi_cutoff = isigi_cutoff
        # Explicit MTZ metadata columns, or None to let MTZLoader select the
        # ones from MTZ_METADATA_KEYS that the file actually carries.
        self.mtz_metadata = mtz_metadata
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = None
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.epochs = epochs
        self.ambiguate = ambiguate
        self.reflections_per_image = reflections_per_image
        self.sample_with_replacement = sample_with_replacement
        # Seeds the shuffle buffers only. None gives a new order every run, as
        # the unseeded tf.data shuffle did. The train/test split and the
        # ambiguation are deterministic regardless.
        self.seed = seed

    def get_config(self):
        conf = {
            'dmin' : self.dmin,
            'wavelength' : self.wavelength,
            'inputs' : self.inputs,
            'cell' : list(self.cell.parameters), 
            'spacegroup' : self.spacegroup.xhm(),
            'num_cpus' : self.num_cpus,
            'separate' : self.separate,
            'wavelength' : self.wavelength,
            'ray_log_level' : self.ray_log_level,
            'test_fraction' : self.test_fraction,
            'num_asus': self.num_asus,
            'separate_friedel_mates' : self.separate_friedel_mates,
            'shuffle_buffer_size' : self.shuffle_buffer_size,
            'batch_size' : self.batch_size,
            'steps_per_epoch' : self.steps_per_epoch,
            'validation_steps' : self.validation_steps,
            'epochs' : self.epochs,
            'ambiguate' : self.ambiguate,
            'mtz_metadata' : self.mtz_metadata,
            'reflections_per_image' : self.reflections_per_image,
            'sample_with_replacement' : self.sample_with_replacement,
            'seed' : self.seed,
        }
        return conf

    @classmethod
    def from_config(cls, conf):
        num_asus = conf.pop('num_asus')
        result = cls(**conf)
        result.num_asus = num_asus
        return result

    @classmethod
    def from_parser(cls, parser):
        return cls(
            inputs = parser.inputs,
            mtz_metadata = _split_metadata_keys(parser.mtz_metadata),
            dmin = parser.dmin,
            cell = parser.cell, 
            spacegroup = parser.space_group,
            num_cpus = parser.num_cpus,
            separate = parser.separate,
            wavelength = parser.wavelength,
            ray_log_level = parser.ray_log_level,
            test_fraction = parser.test_fraction,
            separate_friedel_mates = parser.separate_friedel_mates,
            cell_tol = parser.fractional_cell_tolerance,
            isigi_cutoff = parser.isigi_cutoff,
            shuffle_buffer_size = parser.shuffle_buffer_size,
            batch_size = parser.batch_size,
            steps_per_epoch = parser.steps_per_epoch,
            validation_steps = parser.validation_steps,
            epochs = parser.epochs,
            ambiguate = parser.enable_index_ambiguation,
            reflections_per_image = parser.reflections_per_image,
            sample_with_replacement = not parser.sample_without_replacement,
            seed = getattr(parser, 'seed', None),
        )

    @property
    def cell(self):
        """Unit cell parameters (a, b, c, alpha, beta, gamma)"""
        return self._cell

    @property
    def spacegroup(self):
        """Crystallographic space group"""
        return self._spacegroup

    @spacegroup.setter
    @spacegroupify("sg")
    def spacegroup(self, sg):
        self._spacegroup = sg

    @cell.setter
    @cellify("uc")
    def cell(self, uc):
        self._cell = uc

    def get_dataset(self):
        """
        The images of all inputs, in input order, as an iterable of records.

        Each record is a tuple of seven numpy arrays (see `DataLoader`). The
        loaders are set up here, which settles `cell`, `spacegroup` and
        `num_asus`, but the files are parsed only as the result is iterated,
        and it can be iterated only once.
        """
        asu_id = 0
        if all([_is_stream_file(f) for f in self.inputs]):
            from abismal.io import StreamLoader
            data = []

            for stream_file in self.inputs:
                loader = StreamLoader(
                    stream_file, 
                    cell=self.cell, 
                    dmin=self.dmin, 
                    asu_id=asu_id, 
                    wavelength=self.wavelength,
                    isigi_cutoff=self.isigi_cutoff,
                    cell_tol=self.cell_tol,
                )
                if self.separate:
                    self.num_cpus = 1 #multiprocessing with ray can't be done on multiple files at once
                    asu_id += 1
                if self.cell is None:
                    self.cell = loader.cell
                data.append(loader.get_dataset(
                    num_cpus=self.num_cpus,
                    logging_level=self.ray_log_level,
                ))

        elif all([_is_dials_file(f) for f in self.inputs]):
            # Checked before the loader, not inside it. dxtbx is imported several
            # frames down and after the manager has started work, so a missing
            # DIALS stack otherwise surfaces as a bare ModuleNotFoundError with
            # no hint that the fix is a conda install, or that only these file
            # types need one.
            from abismal.io._dials_check import require_dials

            require_dials([f for f in self.inputs if _is_dials_file(f)])

            from abismal.io import StillsLoader
            expt_files = [f for f in self.inputs if _is_expt_file(f)]
            refl_files = [f for f in self.inputs if _is_refl_file(f)]

            data = []
            if self.separate:
                for expt,refl in zip(expt_files, refl_files):
                    loader = StillsLoader([expt], [refl], self.spacegroup, self.cell, self.dmin, asu_id, cell_tol=self.cell_tol, isigi_cutoff=self.isigi_cutoff)
                    asu_id += 1
                    data.append(loader.get_dataset(num_cpus=self.num_cpus))
            else:
                loader = StillsLoader(
                    expt_files, refl_files, self.spacegroup, self.cell, self.dmin, asu_id=asu_id,
                    cell_tol=self.cell_tol, isigi_cutoff=self.isigi_cutoff,
                )
                data.append(loader.get_dataset(num_cpus=self.num_cpus))
            if self.cell is None:
                self.cell = loader.cell
        elif all([_is_mtz_file(f) for f in self.inputs]):
            from abismal.io import MTZLoader
            data = []
            for mtz in self.inputs:
                loader = MTZLoader(mtz, dmin=self.dmin, cell=self.cell,
                                   spacegroup=self.spacegroup, asu_id=asu_id,
                                   metadata_keys=self.mtz_metadata)
                if self.separate:
                    asu_id += 1
                data.append(loader.get_dataset())
            if self.cell is None:
                self.cell = loader.cell

        else:
            raise ValueError(
                "Couldn't determine input file type. "
                "MTZs, DIALS reflection tables, and CrystFEL streams are supported. "
                "Mixing filetypes is not supported."
            )
        if self.spacegroup is None:
            if hasattr(loader, 'spacegroup'):
                self.spacegroup = loader.spacegroup
            else:
                self.spacegroup = 'P1'

        self.num_asus = asu_id
        if not self.separate:
            self.num_asus = self.num_asus + 1
        if self.separate_friedel_mates:
            self.num_asus = 2

        data = chain.from_iterable(data)
        if self.ambiguate:
            from abismal.io.ambiguation import Ambiguator
            amb = Ambiguator.from_symmetry(self.cell, self.spacegroup)
            data = (amb(i, image) for i,image in enumerate(data))
        return data

    @property
    def repeat_train(self):
        if self.steps_per_epoch is None:
            return False
        return True

    @property
    def repeat_test(self):
        repeat_test = self.validation_steps is not None
        return repeat_test

    def friedelize(self, data):
        """
        Put each reflection in ASU 0 if it is a Friedel plus (or centric) and
        in ASU 1 otherwise, which is what --separate-friedel-mates asks for.
        """
        spacegroup = self.spacegroup
        for image in data:
            hkl = image[1]
            # The TF version looked this up in an anomalous ReciprocalASU. Its
            # flag for a reflection is that of the ASU representative, which
            # shares the reflection's Friedel half, so asking directly is the
            # same thing for every reflection the ASU holds.
            _,isym = rs.utils.hkl_to_asu(hkl, spacegroup)
            is_plus = ((isym % 2) == 1) | rs.utils.is_centric(hkl, spacegroup)
            asu_id = np.where(is_plus, 0, 1).astype(np.int32)[:,None]
            yield (asu_id,) + tuple(image[1:])

    def get_train_test_splits(self, data=None, test_fraction=None, repeat_test=None, repeat_train=None):
        """
        Returns `(train, test)` `BatchStream`s, test None when there is no
        test fraction.

        This returns at once. Parsing starts in a background thread on the
        first iteration of either stream, and training can begin on the first
        images while the rest are parsed. Both streams share one in-memory
        cache of the parsed images, so later passes do not read the files
        again. See `abismal.io.pipeline` for the details.
        """
        if data is None:
            data = self.get_dataset()
        if test_fraction is None:
            test_fraction = self.test_fraction
        if repeat_test is None:
            repeat_test = self.repeat_test
        if repeat_train is None:
            repeat_train = self.repeat_train

        if self.separate_friedel_mates:
            data = self.friedelize(data)

        cache = ImageCache(data, test_fraction=test_fraction)
        # Distinct seeds so the two shuffles are not the same sequence.
        train_seed = test_seed = None
        if self.seed is not None:
            train_seed, test_seed = [self.seed, 0], [self.seed, 1]

        train = BatchStream(
            cache, 'train', batch_size=self.batch_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
            repeat=repeat_train, seed=train_seed,
            reflections_per_image=self.reflections_per_image,
            sample_with_replacement=self.sample_with_replacement,
        )
        test = None
        if test_fraction > 0.:
            test = BatchStream(
                cache, 'test', batch_size=self.batch_size,
                shuffle_buffer_size=self.shuffle_buffer_size,
                repeat=repeat_test, seed=test_seed,
                reflections_per_image=self.reflections_per_image,
                sample_with_replacement=self.sample_with_replacement,
            )
        return train, test

    def get_half_datasets(self):
        return self.get_train_test_splits(test_fraction = 0.5, repeat_test = self.repeat_train)

    def to_file(self, file_name):
        conf = self.get_config()
        with open(file_name, 'w') as out:
            yaml.dump(conf, out)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r') as f:
            conf = yaml.safe_load(f)
        return cls.from_config(conf)

