from reciprocalspaceship.decorators import spacegroupify,cellify
from abismal.io import split_dataset_train_test
import pickle

# TODO: refactor this filetype control flow into abismal.io
_file_endings = {
    'refl' : ('.refl', '.pickle'),
    'expt' : ('.expt', '.json'),
    'stream' : ('.stream',),
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

def _is_dials_file(s):
    return _is_refl_file(s) or _is_expt_file(s)

class DataManager:
    """
    A high-level class for managing I/O of reflection data.
    """
    def __init__(self, inputs, dmin, cell=None, spacegroup=None, 
            num_cpus=None, separate=False, wavelength=None, ray_log_level="ERROR",
            test_fraction=0.):
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
                )
                if self.separate:
                    asu_id += 1
                else:
                    asu_id = 1
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
                    loader = StillsLoader([expt], [refl], self.spacegroup, self.cell, self.dmin, asu_id)
                    asu_id += 1
                    _data = loader.get_dataset()
                    if data is None:
                        data = _data
                    else:
                        data = data.concatenate(_data)
            else:
                loader = StillsLoader(
                    expt_files, refl_files, self.spacegroup, self.cell, self.dmin, asu_id=asu_id
                )
                data = loader.get_dataset()
                asu_id += 1
            if self.cell is None:
                self.cell = loader.cell
        else:
            raise ValueError(
                "Couldn't determine input file type. "
                "DIALS reflection tables and CrystFEL streams are supported. "
                "Mixing filetypes is not supported."
            )
        if self.spacegroup is None:
            if hasattr(loader, 'spacegroup'):
                self.spacegroup = loader.spacegroup
            else:
                self.spacegroup = 'P1'

        self.num_asus = asu_id

        return data

    def get_train_test_splits(self, data=None):
        if data is None:
            data = self.get_dataset()

        # Handle setting up the test fraction, shuffle buffer, batching, etc
        test = None
        if self.test_fraction > 0.:
            train,test = split_dataset_train_test(data, self.test_fraction)
        else:
            train = data
        return train, test

    def to_file(self, file_name):
        with open(file_name, 'wb') as out:
            pickle.dump(self, out)

    def from_file(self, file_name):
        with open(file_name, 'rb') as f:
            dm = pickle.load(file_name)

