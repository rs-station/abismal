"""The framework-free input pipeline in abismal.io.pipeline.

These pin the behaviour abismal had with tf.data: asynchronous parsing with
training starting during the first pass, an in-memory cache replayed on later
passes, a deterministic train/test split, a sliding shuffle buffer placed after
the repeat, and ragged batches of whole images.
"""
import threading
import time
from itertools import islice

import numpy as np
import pytest

from abismal.io.batch import Batch
from abismal.io.pipeline import (
    BatchStream, ImageCache, batched, prefetch, reservoir_shuffle,
)
from abismal.io.utils import index_uniform, is_test_index


pytestmark = pytest.mark.timeout(60)


def image(i, n=None):
    """ A record whose reflections all carry `i`, so images can be told apart """
    if n is None:
        n = 1 + i % 4
    f = np.full((n, 1), i, dtype='float32')
    return (
        np.zeros((n, 1), 'int32'),
        np.full((n, 3), i, 'int32'),
        f, f, np.full((n, 2), i, 'float32'), f, f,
    )


def ids(images):
    return [int(x[1][0, 0]) for x in images]


def batch_ids(batch):
    """ The image ids in a Batch, in order """
    starts = np.r_[0, np.cumsum(np.bincount(batch.image_id, minlength=batch.num_images))[:-1]]
    return [int(batch.hkl[s, 0]) for s in starts]


def slow_source(n, delay, started=None):
    for i in range(n):
        if started is not None:
            started.set()
        time.sleep(delay)
        yield image(i)


# ---------------------------------------------------------------- split

def test_split_is_deterministic():
    a = [is_test_index(i, 0.2) for i in range(5000)]
    b = [is_test_index(i, 0.2) for i in range(5000)]
    assert a == b
    other = [is_test_index(i, 0.2, seed=99) for i in range(5000)]
    assert a != other


def test_split_fraction():
    for frac in (0.05, 0.1, 0.5):
        n = 20_000
        k = sum(is_test_index(i, frac) for i in range(n))
        sd = np.sqrt(n * frac * (1 - frac))
        assert abs(k - n * frac) < 5 * sd
    assert not any(is_test_index(i, 0.) for i in range(1000))


def test_index_uniform_range():
    u = np.array([index_uniform(i, 7) for i in range(10_000)])
    assert (u >= 0).all() and (u < 1).all()
    # roughly uniform
    counts = np.histogram(u, bins=10, range=(0, 1))[0]
    assert counts.min() > 800


def test_cache_split_matches_index():
    cache = ImageCache((image(i) for i in range(500)), test_fraction=0.3)
    train = ids(cache.iterate('train'))
    test = ids(cache.iterate('test'))
    assert sorted(train + test) == list(range(500))
    assert test == [i for i in range(500) if is_test_index(i, 0.3)]
    # in input order
    assert train == sorted(train)


# ---------------------------------------------------------------- cache

def test_cache_replays_from_memory():
    calls = []
    def source():
        for i in range(10):
            calls.append(i)
            yield image(i)
    cache = ImageCache(source())
    first = ids(cache.iterate('train'))
    second = ids(cache.iterate('train'))
    assert first == second == list(range(10))
    assert calls == list(range(10))


def test_cache_repeat():
    cache = ImageCache(image(i) for i in range(5))
    out = ids(islice(cache.iterate('train', repeat=True), 17))
    assert out == (list(range(5)) * 4)[:17]


def test_cache_repeat_of_empty_split_ends():
    cache = ImageCache(image(i) for i in range(5))
    assert list(cache.iterate('test', repeat=True)) == []


def test_epoch_one_overlaps_parsing():
    """ A batch is ready long before the slow source is exhausted """
    cache = ImageCache(slow_source(50, 0.02))
    stream = BatchStream(cache, batch_size=2, shuffle_buffer_size=0)
    t = time.time()
    first = next(iter(stream))
    assert isinstance(first, Batch)
    assert not cache.done
    assert time.time() - t < 0.5
    assert len(cache) < 50

    # And the rest of the first pass still arrives, in full
    rest = list(stream)
    assert sum(b.num_images for b in rest) == 50
    assert cache.done


def test_epoch_one_overlap_with_prefetch_and_shuffle():
    cache = ImageCache(slow_source(60, 0.02))
    stream = BatchStream(cache, batch_size=4, shuffle_buffer_size=8, repeat=True, seed=0)
    with stream.iterate(prefetch=2) as it:
        b = next(it)
        assert b.num_images == 4
        assert not cache.done


def test_get_train_test_splits_returns_before_parsing(conventional_mtz):
    from abismal.io.manager import DataManager
    started = threading.Event()
    dm = DataManager([conventional_mtz], dmin=4., batch_size=2)
    train, test = dm.get_train_test_splits(data=slow_source(20, 0.05, started))
    assert test is None
    assert not started.is_set()
    next(iter(train))
    assert started.is_set()
    assert not train.done


def test_concurrent_readers():
    """ Several readers follow the same growing cache independently """
    cache = ImageCache(slow_source(40, 0.005), test_fraction=0.25)
    results = {}
    def read(name, split):
        results[name] = ids(cache.iterate(split))
    threads = [
        threading.Thread(target=read, args=(f"{s}{k}", s))
        for s in ('train', 'test') for k in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    expected_test = [i for i in range(40) if is_test_index(i, 0.25)]
    expected_train = [i for i in range(40) if not is_test_index(i, 0.25)]
    for k in range(3):
        assert results[f"train{k}"] == expected_train
        assert results[f"test{k}"] == expected_test


class Boom(RuntimeError):
    pass


def failing_source(n_good):
    for i in range(n_good):
        yield image(i)
    raise Boom("parse failure")


def test_producer_exception_reaches_consumer():
    cache = ImageCache(failing_source(5))
    with pytest.raises(Boom):
        list(cache.iterate('train'))
    # And keeps doing so, for every reader
    with pytest.raises(Boom):
        list(cache.iterate('test'))
    assert cache.done


def test_producer_exception_through_prefetch():
    stream = BatchStream(ImageCache(failing_source(5)), batch_size=2, repeat=True)
    with pytest.raises(Boom):
        with stream.iterate(prefetch=2) as it:
            for _ in it:
                pass


# ---------------------------------------------------------------- shuffle

def test_shuffle_is_a_permutation():
    rng = np.random.default_rng(0)
    out = list(reservoir_shuffle(range(1000), 100, rng))
    assert sorted(out) == list(range(1000))
    assert out != list(range(1000))


@pytest.mark.parametrize("size", [0, 1, None, -5])
def test_shuffle_small_buffer_is_identity(size):
    assert list(reservoir_shuffle(range(100), size)) == list(range(100))


def test_shuffle_is_local():
    """ A sliding buffer of K only lets an element move ~K places earlier """
    K = 10
    out = list(reservoir_shuffle(range(1000), K, np.random.default_rng(1)))
    for position, x in enumerate(out):
        # x was read once at least x - K + 1 elements had been emitted
        assert position >= x - K
    # ...but later is unbounded and a large buffer mixes thoroughly
    big = list(reservoir_shuffle(range(1000), 2000, np.random.default_rng(1)))
    assert abs(np.corrcoef(big, range(1000))[0, 1]) < 0.2


def test_shuffle_first_element_is_uniform_over_buffer():
    """ tf.data semantics: the first output is uniform over the first K inputs """
    K = 5
    rng = np.random.default_rng(2)
    first = [next(reservoir_shuffle(range(100), K, rng)) for _ in range(5000)]
    counts = np.bincount(first, minlength=K)
    assert len(counts) == K
    assert counts.min() > 800


def test_shuffle_seed():
    a = list(reservoir_shuffle(range(100), 50, np.random.default_rng(3)))
    b = list(reservoir_shuffle(range(100), 50, np.random.default_rng(3)))
    assert a == b


def test_stream_reshuffles_every_pass():
    cache = ImageCache(image(i) for i in range(50))
    stream = BatchStream(cache, batch_size=1, shuffle_buffer_size=50, seed=0)
    passes = [[batch_ids(b)[0] for b in stream] for _ in range(3)]
    for p in passes:
        assert sorted(p) == list(range(50))
    assert passes[0] != passes[1] != passes[2]

    # Seeded streams are reproducible
    again = BatchStream(cache, batch_size=1, shuffle_buffer_size=50, seed=0)
    assert [batch_ids(b)[0] for b in again] == passes[0]


def test_shuffle_buffer_spans_passes_when_repeating():
    """ The buffer sits after the repeat, as in tf.data, so passes mix """
    n = 20
    cache = ImageCache(image(i) for i in range(n))
    stream = BatchStream(cache, batch_size=1, shuffle_buffer_size=n, repeat=True, seed=4)
    out = [batch_ids(b)[0] for b in islice(iter(stream), 20 * n)]
    windows = [sorted(out[k:k + n]) for k in range(0, len(out), n)]
    # If passes did not mix, every window of n outputs would be one exact pass
    assert any(w != list(range(n)) for w in windows)
    # Nothing is lost or duplicated overall
    counts = np.bincount(out, minlength=n)
    assert counts.max() - counts.min() <= 2 * n


# ---------------------------------------------------------------- batching

def test_batched_sizes_and_remainder():
    batches = list(batched((image(i) for i in range(10)), 4))
    assert [b.num_images for b in batches] == [4, 4, 2]
    assert [batch_ids(b) for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    for b in batches:
        lengths = np.bincount(b.image_id, minlength=b.num_images)
        assert list(lengths) == [1 + i % 4 for i in batch_ids(b)]
        assert len(b.hkl) == len(b.image_id) == lengths.sum()


def test_batched_rejects_zero():
    with pytest.raises(ValueError):
        list(batched([image(0)], 0))


# ---------------------------------------------------------------- prefetch

def test_prefetch_preserves_order_and_applies_fn():
    out = list(prefetch(range(100), fn=lambda x: x * 2, depth=3))
    assert out == [2 * x for x in range(100)]


def test_prefetch_propagates_fn_errors():
    def fn(x):
        if x == 5:
            raise Boom
        return x
    it = prefetch(range(10), fn=fn)
    with pytest.raises(Boom):
        list(it)


def test_prefetch_early_stop_does_not_hang():
    produced = []
    def endless():
        i = 0
        while True:
            produced.append(i)
            yield i
            i += 1
    before = threading.active_count()
    p = prefetch(endless(), depth=2)
    assert next(p) == 0
    thread = p._thread
    p.close()
    thread.join(5)
    assert not thread.is_alive()
    # It stopped pulling from upstream promptly
    n = len(produced)
    time.sleep(0.3)
    assert len(produced) == n
    assert threading.active_count() <= before
    with pytest.raises(StopIteration):
        next(p)


def test_prefetch_garbage_collected_stops():
    def endless():
        i = 0
        while True:
            yield i
            i += 1
    p = prefetch(endless(), depth=2)
    next(p)
    thread = p._thread
    del p
    thread.join(5)
    assert not thread.is_alive()


def test_stream_iterate_transfer():
    cache = ImageCache(image(i) for i in range(10))
    stream = BatchStream(cache, batch_size=3)
    seen = []
    def transfer(batch):
        seen.append(threading.current_thread().name)
        return batch.map(lambda a: a.copy())
    out = list(stream.iterate(transfer=transfer, prefetch=2))
    assert [b.num_images for b in out] == [3, 3, 3, 1]
    assert all(name == "abismal-prefetch" for name in seen)
    assert batch_ids(out[0]) == [0, 1, 2]

    inline = list(stream.iterate(transfer=transfer, prefetch=0))
    assert [b.num_images for b in inline] == [3, 3, 3, 1]


def test_sample_reflections_fixes_the_shape():
    from abismal.io.pipeline import sample_reflections
    rng = np.random.default_rng(0)
    for n in (3, 10, 50):
        image = (np.arange(n)[:, None], np.arange(3 * n).reshape(n, 3))
        for replace in (True, False):
            out = sample_reflections(image, 20, rng, replace)
            assert out[0].shape == (20, 1) and out[1].shape == (20, 3)
            # rows stay intact
            assert (out[1][:, 0] == 3 * out[0][:, 0]).all()
    # without replacement, an image with enough reflections has no repeats
    out = sample_reflections((np.arange(50)[:, None],), 20, rng, replace=False)
    assert len(np.unique(out[0])) == 20


def test_fixed_shape_batches_are_all_full():
    from abismal.io.pipeline import ImageCache, BatchStream
    rng = np.random.default_rng(1)
    images = []
    for i in range(23):
        n = int(rng.integers(5, 40))
        images.append(tuple(np.zeros((n, w), np.float32) for w in (1, 3, 1, 1, 2, 1, 1)))
    stream = BatchStream(ImageCache(images), batch_size=5, reflections_per_image=8, seed=0)
    batches = list(stream)
    assert len(batches) == 4  # the 3 leftover images are dropped
    for b in batches:
        assert b.num_images == 5 and len(b.iobs) == 40
        assert (b.image_id == np.repeat(np.arange(5), 8)).all()
