import pytest 
import reciprocalspaceship as rs
from abismal.io import MTZLoader, StillsLoader



def test_mtz_loader():
    loader = MTZLoader("test_data.mtz")
    ds = loader.get_dataset()
