import pytest 
import reciprocalspaceship as rs
from abismal.io import MTZLoader, StillsLoader



def test_mtz_loader(conventional_mtz):
    loader = MTZLoader(conventional_mtz)
    ds = loader.get_dataset()

    #Test unbatched iteration
    for i in ds:
        break

    #Test batched iteration
    for i in ds.batch(2):
        break

@pytest.mark.xfail
def test_stills_loader(stills_expt, stills_refl):
    loader = StillsLoader(
        [
            stills_expt,
            stills_expt,
            stills_expt,
        ],
        [
            stills_refl,
            stills_refl,
            stills_refl,
        ],
    )
    ds = loader.get_dataset()

    #Test unbatched iteration
    for i in ds:
        break

    #Test batched iteration
    for i in ds.batch(2):
        break

