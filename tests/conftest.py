import pytest
from os.path import dirname, abspath, join



@pytest.fixture
def conventional_mtz():
    datapath = "data/conventional.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def stills_refl():
    datapath = "data/stills.refl"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def stills_expt():
    datapath = "data/stills.expt"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def stills_stream():
    datapath = "data/crystfel.stream"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

