import os

# Hide the GPU from every test process, before anything initializes CUDA.
#
# addopts carries `-n auto`, so pytest-xdist starts one worker per core -- 16 on
# this machine. torch allocates GPU memory lazily, so importing it no longer
# takes the whole card the way TensorFlow did, but any test that lands on the
# GPU still costs that worker a CUDA context of several hundred MB, sixteen
# times over, and makes results depend on which device a test happened to pick:
# `--gpu-id` defaults to 0, so a CLI test trains on the card whenever one is
# visible. Running everything on the CPU keeps the suite deterministic and the
# card free.
#
# A worker that hangs while holding the card is worse. One ran for 46 hours,
# keeping 46 GB of GPU and -- through file handles on already-unlinked ray spill
# files -- 264 GB of disk that `du` could not see, until the filesystem hit 100%.
#
# Set here rather than per-file because CUDA reads the variable once, when it
# initializes, and conftest is imported before the test modules.
#
# ABISMAL_TEST_GPU=1 opts back in, for deliberately testing device placement.
# Use it with `-n 0`; with xdist every worker contends for the same card.
if not os.environ.get("ABISMAL_TEST_GPU"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import pytest

# Test modules import the model builders in model_helpers.py with `from model_helpers import ...`,
# which needs this directory importable whichever worker collects them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


