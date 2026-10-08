import pytest
import numpy as np
import reciprocalspaceship as rs
from abismal.io import MTZLoader, StillsLoader, StreamLoader
from abismal.io.batch import Batch


RECORD_DTYPES = (np.int32, np.int32, np.float32, np.float32, np.float32, np.float32, np.float32)
RECORD_WIDTHS = (1, 3, 1, 1, None, 1, 1)


def check_record(record, metadata_length=None):
    """A per-image record: seven arrays, one row per reflection."""
    assert isinstance(record, tuple)
    assert len(record) == 7
    n = len(record[0])
    assert n >= 1
    for a, dtype, width in zip(record, RECORD_DTYPES, RECORD_WIDTHS):
        assert isinstance(a, np.ndarray)
        assert a.dtype == dtype
        assert a.ndim == 2
        assert a.shape[0] == n
        if width is not None:
            assert a.shape[1] == width
    if metadata_length is not None:
        assert record[4].shape[1] == metadata_length


def test_mtz_loader(conventional_mtz):
    loader = MTZLoader(conventional_mtz, asu_id=3)
    records = list(loader.get_dataset())

    for r in records:
        check_record(r, loader.metadata_length)
        assert (r[0] == 3).all()
        assert (r[3] == 1.).all()

    # One record per batch, holding exactly that batch's reflections
    ds = rs.read_mtz(conventional_mtz)
    ds = ds[~ds.label_absences().ABSENT]
    counts = ds.groupby(loader.batch_key).size().to_numpy()
    assert len(records) == len(counts)
    assert [len(r[0]) for r in records] == list(counts)
    intensities = np.sort(np.concatenate([r[5] for r in records]).ravel())
    assert np.allclose(intensities, np.sort(ds[loader.intensity_key].to_numpy('float32')))


def test_mtz_loader_dmin(conventional_mtz):
    records = list(MTZLoader(conventional_mtz, dmin=4.).get_dataset())
    d = np.concatenate([r[2] for r in records])
    assert d.min() >= 4.
    assert len(records) > 0


def test_stream_loader(stills_stream):
    loader = StreamLoader(stills_stream, dmin=4., asu_id=1, wavelength=1.)
    records = list(loader.get_dataset(num_cpus=1))
    assert len(records) == 3
    for r in records:
        check_record(r, 12)
        assert (r[0] == 1).all()
        assert (r[2] >= 4.).all()
        assert np.allclose(r[3], 1.)


@pytest.mark.timeout(300)
def test_stream_loader_ray_matches_serial(stills_stream):
    """ Parsing with ray must give the same images, in the same order """
    loader = StreamLoader(stills_stream, dmin=4.)
    serial = list(loader.get_dataset(num_cpus=1))
    parallel = list(loader.get_dataset(num_cpus=2, logging_level="ERROR"))
    assert len(serial) == len(parallel)
    for a, b in zip(serial, parallel):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_stills_loader(stills_expt, stills_refl):
    # Needs DIALS. The documented install always has it, and CI installs it, but
    # skip rather than fail for a checkout without it. This carried a bare xfail
    # for the same reason, which then reported an unexpected pass everywhere the
    # install is correct.
    pytest.importorskip("dxtbx")
    loader = StillsLoader(
        [stills_expt] * 3,
        [stills_refl] * 3,
    )
    records = list(loader.get_dataset())
    assert len(records) > 0
    assert len(records) % 3 == 0
    for r in records:
        # 9 columns: s1 (3), background and its sigma (2), delpsical (1) and dQ (3).
        # (StillsLoader.metadata_length says 6; nothing downstream reads it.)
        check_record(r, 9)

    # The three copies are the same images, in file order
    k = len(records) // 3
    for a, b in zip(records[:k], records[k:2*k]):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_batch_collate():
    rng = np.random.default_rng(0)
    lengths = [3, 1, 5]
    images = []
    for n in lengths:
        images.append((
            np.zeros((n, 1), 'int32'),
            rng.integers(-5, 5, (n, 3)).astype('int32'),
            rng.random((n, 1), 'float32'),
            np.ones((n, 1), 'float32'),
            rng.random((n, 4), 'float32'),
            rng.random((n, 1), 'float32'),
            rng.random((n, 1), 'float32'),
        ))
    batch = Batch.collate(images)
    assert batch.num_images == 3
    assert batch.image_id.dtype == np.int64
    assert list(batch.image_id) == [0, 0, 0, 1, 2, 2, 2, 2, 2]
    assert np.array_equal(np.bincount(batch.image_id), lengths)
    assert len(batch.inputs) == 7
    for field, parts in zip(batch.inputs, zip(*images)):
        assert np.array_equal(field, np.concatenate(parts))

    doubled = batch.map(lambda a: a * 2)
    assert isinstance(doubled, Batch)
    assert doubled.num_images == 3
    assert np.array_equal(doubled.image_id, 2 * batch.image_id)


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
@pytest.mark.timeout(120)
def test_datamanager(
        file_format, double, separate, stills_expt, stills_refl,
        stills_stream, conventional_mtz, test_fraction, batch_size=1
    ):
    if file_format == 'mtz':
        inputs = [conventional_mtz]
    elif file_format == 'stream':
        inputs = [stills_stream]
    elif file_format == 'dials':
        pytest.importorskip("dxtbx")
        inputs = [stills_expt, stills_refl]
    if double:
        inputs = inputs + inputs

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

    expected_asus = 2 if (double and separate) else 1
    assert dm.num_asus == expected_asus

    batches = list(train)
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, Batch)
        assert len(batch.inputs) == 7
        assert batch.num_images == batch_size
        assert batch.asu_id.max() < dm.num_asus

    num_test = 0 if test is None else sum(b.num_images for b in test)
    num_train = sum(b.num_images for b in batches)
    assert num_train + num_test == len(train.cache)


@pytest.mark.timeout(120)
def test_separate_friedel_mates(conventional_mtz):
    from abismal.io.manager import DataManager
    dm = DataManager([conventional_mtz], dmin=4., separate_friedel_mates=True, batch_size=1)
    train, _ = dm.get_train_test_splits()
    assert dm.num_asus == 2
    for batch in train:
        hkl = batch.hkl
        _, isym = rs.utils.hkl_to_asu(hkl, dm.spacegroup)
        centric = rs.utils.is_centric(hkl, dm.spacegroup)
        plus = (isym % 2 == 1) | centric
        assert np.array_equal(batch.asu_id.ravel(), np.where(plus, 0, 1))
        assert set(np.unique(batch.asu_id)) <= {0, 1}
