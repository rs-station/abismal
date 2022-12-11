import pytest 
import reciprocalspaceship as rs
from abismal.io import MTZLoader, StillsLoader



def test_mtz_loader():
    loader = MTZLoader("conventional.mtz")
    ds = loader.get_dataset()

@pytest.mark.xfail
def test_stills_loader():
    loader = StillsLoader(
        [
            "stills.expt",
            "stills.expt",
            "stills.expt",
        ],
        [
            "stills.refl",
            "stills.refl",
            "stills.refl",
        ],
    )
    ds = loader.get_dataset()

