import os

# Hide the GPU from every test process, before anything imports TensorFlow.
#
# addopts carries `-n auto`, so pytest-xdist starts one worker per core -- 16 on
# this machine -- and each one imports TensorFlow and takes the card's entire
# default allocation. set_memory_growth is called by abismal.io.set_gpu on the
# CLI path and never in a test, so nothing shares. The first worker wins the
# card and the rest die on `cudaSetDevice() ... out of memory`, which reads as a
# real failure in code that has nothing to do with devices: 64 tests pass under
# `-n 0` and 7 fail under `-n auto`, same card, same tests.
#
# A worker that hangs while holding the card is worse. One ran for 46 hours,
# keeping 46 GB of GPU and -- through file handles on already-unlinked ray spill
# files -- 264 GB of disk that `du` could not see, until the filesystem hit 100%.
#
# Set here rather than per-file because it has to beat the first `import
# tensorflow` in any worker, and conftest is imported before the test modules.
# test_cli, test_cchalf and test_transform_choices each call
# set_visible_devices([], 'GPU') already, but `--gpu-id` defaults to 0 and
# set_gpu turns the device back on inside main(), so whether a CLI test touched
# the GPU depended on what had run before it in the same process.
#
# ABISMAL_TEST_GPU=1 opts back in, for deliberately testing device placement.
# Use it with `-n 0`; with xdist it reproduces exactly the problem above.
if not os.environ.get("ABISMAL_TEST_GPU"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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

