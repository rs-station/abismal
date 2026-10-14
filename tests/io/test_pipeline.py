"""The framework-free input pipeline in abismal.io.pipeline.

These pin the behaviour abismal had with tf.data: asynchronous parsing with
training starting during the first pass, an in-memory cache replayed on later
passes, a deterministic train/test split, and ragged batches of whole images.
Shuffling departs from tf.data on purpose: the first pass follows the parser,
and every pass that starts with all images parsed is a fresh permutation.
"""
import threading
import time
from itertools import islice

import numpy as np
import pytest

from abismal.io.batch import Batch
from abismal.io.pipeline import (
    BatchStream, ImageCache, batched, prefetch,
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
    stream = BatchStream(cache, batch_size=2, shuffle=True)
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
    stream = BatchStream(cache, batch_size=4, shuffle=True, repeat=True, seed=0)
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

def first_ids(stream, count=None):
    it = iter(stream) if count is None else islice(iter(stream), count)
    return [batch_ids(b)[0] for b in it]


def test_first_pass_follows_the_parser_in_input_order():
    """ A pass that starts while parsing is under way is in input order """
    cache = ImageCache(slow_source(30, 0.005))
    stream = BatchStream(cache, batch_size=1, shuffle=True, seed=0)
    assert first_ids(stream) == list(range(30))


def test_passes_over_a_complete_cache_are_fresh_permutations():
    n = 50
    cache = ImageCache(image(i) for i in range(n))
    cache.wait()
    stream = BatchStream(cache, batch_size=1, shuffle=True, seed=0)
    passes = [first_ids(stream) for _ in range(3)]
    for p in passes:
        assert sorted(p) == list(range(n))   # every image exactly once
    assert passes[0] != list(range(n))
    assert passes[0] != passes[1] != passes[2]

    # Seeded streams are reproducible
    again = BatchStream(cache, batch_size=1, shuffle=True, seed=0)
    assert first_ids(again) == passes[0]


def test_repeating_stream_permutes_every_pass_after_the_first():
    """ With repeat=True: input order while parsing, then one permutation per pass """
    n = 20
    cache = ImageCache(slow_source(n, 0.005))
    stream = BatchStream(cache, batch_size=1, shuffle=True, repeat=True, seed=4)
    out = first_ids(stream, 6 * n)
    assert out[:n] == list(range(n))
    windows = [out[k:k + n] for k in range(n, len(out), n)]
    for w in windows:
        assert sorted(w) == list(range(n))   # each later pass sees every image once
    assert len({tuple(w) for w in windows}) == len(windows)


def test_no_image_repeats_within_a_batch():
    """ Unlike a buffer larger than the data, a permutation never repeats an image """
    n = 40
    cache = ImageCache(image(i) for i in range(n))
    cache.wait()
    stream = BatchStream(cache, batch_size=n, shuffle=True, repeat=True, seed=1)
    for b in islice(iter(stream), 10):
        ids = batch_ids(b)
        assert len(ids) == len(set(ids)) == n


def test_shuffle_off_keeps_input_order():
    cache = ImageCache(image(i) for i in range(25))
    cache.wait()
    stream = BatchStream(cache, batch_size=1, shuffle=False)
    assert first_ids(stream) == list(range(25))
    assert first_ids(stream) == list(range(25))


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
