from reciprocalspaceship.decorators import spacegroupify,cellify
from abismal.io import split_dataset_train_test
import reciprocalspaceship as rs
import tensorflow as tf
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

class DataManager:
    """
    A high-level class for managing I/O of reflection data.
    """
    def __init__(self, inputs, dmin, cell=None, spacegroup=None, 
            num_cpus=None, separate=False, wavelength=None, ray_log_level="ERROR",
            test_fraction=0., separate_friedel_mates=False, cell_tol=None, isigi_cutoff=None):
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
        asu_id = 0
        if all([_is_stream_file(f) for f in self.inputs]):
            from abismal.io import StreamLoader
            data = None

            for stream_file in self.inputs:
                loader = StreamLoader(
                    stream_file, 
                    cell=self.cell, 
                    dmin=self.dmin, 
                    asu_id=asu_id, 
                    wavelength=self.wavelength,
                    isigi_cutoff=self.isigi_cutoff,
                )
                if self.separate:
                    asu_id += 1
                if self.cell is None:
                    self.cell = loader.cell
                _data = loader.get_dataset(
                    num_cpus=self.num_cpus,
                    logging_level=self.ray_log_level,
                )
                if data is None:
                    data = _data
                else:
                    data = data.concatenate(_data)

        elif all([_is_dials_file(f) for f in self.inputs]):
            from abismal.io import StillsLoader
            expt_files = [f for f in self.inputs if _is_expt_file(f)]
            refl_files = [f for f in self.inputs if _is_refl_file(f)]

            data = None
            if self.separate:
                for expt,refl in zip(expt_files, refl_files):
                    loader = StillsLoader([expt], [refl], self.spacegroup, self.cell, self.dmin, asu_id, cell_tol=self.cell_tol, isigi_cutoff=self.isigi_cutoff)
                    asu_id += 1
                    _data = loader.get_dataset(num_cpus=self.num_cpus)
                    if data is None:
                        data = _data
                    else:
                        data = data.concatenate(_data)
            else:
                loader = StillsLoader(
                    expt_files, refl_files, self.spacegroup, self.cell, self.dmin, asu_id=asu_id,
                    cell_tol=self.cell_tol, isigi_cutoff=self.isigi_cutoff,
                )
                data = loader.get_dataset(num_cpus=self.num_cpus)
            if self.cell is None:
                self.cell = loader.cell
        elif all([_is_mtz_file(f) for f in self.inputs]):
            from abismal.io import MTZLoader
            data = None
            for mtz in self.inputs:
                loader = MTZLoader(mtz, dmin=self.dmin, cell=self.cell, spacegroup=self.spacegroup, asu_id=asu_id)
                if self.separate:
                    asu_id += 1
                _data = loader.get_dataset()
                if data is None:
                    data = _data
                else:
                    data = data.concatenate(_data)
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

        return data

    def get_train_test_splits(self, data=None):
        if data is None:
            data = self.get_dataset()

        # Handle setting up the test fraction, shuffle buffer, batching, etc
        test = None
        if self.separate_friedel_mates:
            from abismal.symmetry import ReciprocalASU
            rasu = ReciprocalASU(self.cell, self.spacegroup, self.dmin, anomalous=True)
            _,isym = rs.utils.hkl_to_asu(rasu.Hunique, self.spacegroup)
            centric = rs.utils.is_centric(rasu.Hunique, self.spacegroup)
            fplus = (isym % 2) == 1
            fplus = fplus | centric
            def friedelize_datum(x, y):
                asu_id, hkl = x[:2]
                is_plus = rasu.gather(fplus, hkl)
                asu_id = tf.where(is_plus[...,None], 0, 1)
                friedelized = (
                    (asu_id,) + x[1:],
                    y,
                )
                return friedelized
            data = data.map(friedelize_datum)
        if self.test_fraction > 0.:
            train,test = split_dataset_train_test(data, self.test_fraction)
        else:
            train = data
        return train, test

    def to_file(self, file_name):
        conf = self.get_config()
        with open(file_name, 'w') as out:
            yaml.dump(conf, out)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r') as f:
            conf = yaml.safe_load(f)
        return cls.from_config(conf)

