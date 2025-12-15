import pytest
from os.path import dirname, abspath, join



@pytest.fixture
def conventional_mtz():
    datapath = "io/conventional.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def stills_refl():
    datapath = "io/stills.refl"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def stills_expt():
    datapath = "io/stills.expt"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def stills_stream():
    datapath = "io/crystfel.stream"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

