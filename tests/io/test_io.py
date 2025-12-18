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


@pytest.mark.parametrize(
    'file_format',
    ['stream', 'dials', 'mtz'],
)
@pytest.mark.parametrize(
    ('double', 'separate'),(
        (True, True),
        (True, False),
        (False, False),
    )
)
@pytest.mark.parametrize(
    'test_fraction',
    [0.0, 0.1],
)
def test_datamanager(
        file_format, double, separate, stills_expt, stills_refl, 
        stills_stream, conventional_mtz, test_fraction, batch_size=1
    ):
    if file_format == 'mtz':
        inputs = [conventional_mtz]
    elif file_format == 'stream':
        inputs = [stills_stream]
    elif file_format == 'dials':
        inputs = [stills_expt, stills_refl]
    if double:
        inpts = inputs + inputs

    kwargs = {
        'dmin': 4.,
        'cell': None,  #infer
        'spacegroup': None,  #infer
        'num_cpus': 1,  #testing. no mp.
        'separate': False,
        'wavelength': 1., #doesn't matter for execution
        'test_fraction': test_fraction,
        'separate_friedel_mates': False, #this will be a separate test
        'cell_tol': None, #notest
        'isigi_cutoff': None, 
        'shuffle_buffer_size': 0,  #Tested in e2e
        'batch_size': batch_size, 
        'steps_per_epoch': None, 
        'validation_steps': None,
        'epochs': 30,
    }

    if double and separate:
        kwargs['separate'] = True


    from abismal.io.manager import DataManager
    dm = DataManager(inputs, **kwargs)
    train,test = dm.get_train_test_splits()
    if test_fraction == 0.:
        assert test is None
    else:
        assert test is not None

    for x,y in train:
        assert len(x) == 7
        assert len(y) == 1
        assert x[0].shape[0] == batch_size

