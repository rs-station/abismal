"""The streaming input pipeline: cache, split, repeat, shuffle, batch, prefetch.

This reproduces what abismal built from tf.data,

    from_generator -> cache -> train/test filter -> repeat -> shuffle(K)
                   -> ragged_batch(B) -> prefetch

without a framework. The properties that matter to training are kept:

  - Parsing is asynchronous. `ImageCache` drains the loaders in a background
    thread, and the first pass over the data follows the cache as it grows, so
    training starts on the first images while the rest are still being parsed.
  - Parsed images are kept in memory, and every later pass replays them.
  - Train/test membership is a fixed function of each image's position in the
    input (`abismal.io.utils.is_test_index`).
  - `reservoir_shuffle` has the semantics of tf.data's `shuffle`, and sits
    after the repeat, so with `repeat=True` the buffer spans pass boundaries.
  - Batches are B whole images (`Batch.collate`); the last one may be short.
  - `prefetch` runs the whole chain, plus an optional device transfer, in a
    background thread.
"""
import queue
import threading
from itertools import islice

import numpy as np

from abismal.io.batch import Batch
from abismal.io.utils import is_test_index


_END = object()


class ImageCache:
    """Drain an iterable of images once, in a background thread, into memory.

    Element `i` of `source` goes to the 'test' split if
    `is_test_index(i, test_fraction, split_seed)` and to 'train' otherwise.
    Any number of readers may follow either split concurrently (see
    `iterate`); each blocks only when it has caught up with the producer.

    The producer starts on the first read, or on `start()`. An exception
    raised by `source` is re-raised in every reader that asks for more data,
    rather than ending the splits early or leaving readers waiting.
    """
    splits = ('train', 'test')

    def __init__(self, source, test_fraction=0., split_seed=1234):
        self.source = source
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self._images = {k: [] for k in self.splits}
        self._cond = threading.Condition()
        self._thread = None
        self._done = False
        self._error = None

    def start(self):
        with self._cond:
            if self._thread is None:
                # Daemon, so an interpreter that exits mid-parse is not held
                # open by a producer nobody is reading.
                self._thread = threading.Thread(
                    target=self._produce, name="abismal-image-cache", daemon=True,
                )
                self._thread.start()
        return self

    def _produce(self):
        try:
            for i, image in enumerate(self.source):
                split = 'test' if is_test_index(i, self.test_fraction, self.split_seed) else 'train'
                with self._cond:
                    self._images[split].append(image)
                    self._cond.notify_all()
        except BaseException as err:
            with self._cond:
                self._error = err
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    @property
    def done(self):
        """True once the source is exhausted (or failed)."""
        with self._cond:
            return self._done

    def __len__(self):
        with self._cond:
            return sum(len(v) for v in self._images.values())

    def num_images(self, split):
        """How many images of `split` have been parsed so far."""
        with self._cond:
            return len(self._images[split])

    def wait(self, timeout=None):
        """Block until the source is exhausted; return `done`."""
        self.start()
        with self._cond:
            self._cond.wait_for(lambda: self._done, timeout)
            if self._error is not None:
                raise self._error
            return self._done

    def get(self, split, position):
        """Image `position` of `split`, waiting for it if it is not parsed yet.

        Returns the module sentinel `_END` once the source is exhausted and
        `position` is past the end of the split.
        """
        self.start()
        images = self._images[split]
        with self._cond:
            while True:
                if self._error is not None:
                    raise self._error
                if position < len(images):
                    return images[position]
                if self._done:
                    return _END
                self._cond.wait()

    def first_image(self):
        """The first parsed image of either split, waiting only for that one.

        The model needs the metadata width before it can be built, and this is
        the earliest point it is known. Raises if the source produced nothing.
        """
        self.start()
        with self._cond:
            self._cond.wait_for(
                lambda: self._done or any(len(v) for v in self._images.values())
            )
            if self._error is not None:
                raise self._error
            for split in self.splits:
                if len(self._images[split]) > 0:
                    return self._images[split][0]
        raise ValueError("No images were read from the input files")

    def iterate(self, split, repeat=False):
        """Yield the images of `split` in input order.

        The first pass follows the producer and so may wait for parsing; once
        it is done, passes replay from memory. With `repeat=True` the passes
        never end, unless the split is empty.
        """
        if split not in self.splits:
            raise ValueError(f"split must be one of {self.splits}, not {split!r}")
        while True:
            position = 0
            while True:
                image = self.get(split, position)
                if image is _END:
                    break
                yield image
                position += 1
            if not repeat or position == 0:
                return


def reservoir_shuffle(iterable, buffer_size, rng=None):
    """Shuffle a stream through a sliding buffer, as tf.data's `shuffle` does.

    The buffer is filled with the first `buffer_size` elements. Then each
    output is a uniformly chosen buffer slot, refilled from the input. When the
    input runs out, the remaining buffer is drained in random order. An element
    can therefore move at most about `buffer_size` places earlier, and a buffer
    at least as large as the data gives a uniform permutation.

    `buffer_size <= 0` (or 1) passes the input through unchanged.
    `rng` is a `numpy.random.Generator`; None draws a fresh, unseeded one.
    """
    if buffer_size is None or buffer_size <= 1:
        yield from iterable
        return
    if rng is None:
        rng = np.random.default_rng()
    it = iter(iterable)
    buffer = list(islice(it, buffer_size))
    while buffer:
        j = int(rng.integers(len(buffer)))
        item = buffer[j]
        # Emit before reading the next element, as tf.data does, so the
        # consumer is never held up by the producer for an element it could
        # already have had.
        yield item
        try:
            buffer[j] = next(it)
        except StopIteration:
            buffer[j] = buffer[-1]
            buffer.pop()


def batched(iterable, batch_size, drop_remainder=False):
    """Collate consecutive images into `Batch`es of `batch_size` images.

    The last batch holds the remainder, like tf.data's `ragged_batch` with
    `drop_remainder=False`, unless `drop_remainder` is set.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    it = iter(iterable)
    while True:
        images = list(islice(it, batch_size))
        if not images:
            return
        if drop_remainder and len(images) < batch_size:
            return
        yield Batch.collate(images)


def sample_reflections(image, k, rng, replace=True):
    """
    `k` of an image's reflections, drawn at random.

    With `replace=True` they are drawn with replacement. Otherwise they are
    drawn without replacement from images holding at least `k`, and smaller
    images are filled up to `k` with repeats. Either way every image comes out
    the same size, so batches of them have a fixed shape.
    """
    n = len(image[0])
    if replace:
        idx = rng.integers(0, n, size=k)
    elif n >= k:
        idx = rng.choice(n, size=k, replace=False)
    else:
        idx = np.concatenate([rng.permutation(n), rng.integers(0, n, size=k - n)])
    return tuple(a[idx] for a in image)


class prefetch:
    """Iterate `iterable` in a background thread, `depth` items ahead.

    The background thread also applies `fn` to every item, if given. That is
    where the trainer puts its host-to-device copy, so it overlaps the
    previous training step along with the shuffling and collating upstream.

    An exception raised upstream or by `fn` is re-raised from `__next__`.
    Stopping early is safe: `close()` (also called on garbage collection and
    when leaving a `with` block) tells the thread to stop. The thread only
    blocks with a timeout, so it notices within `poll` seconds, except while
    it is itself waiting on upstream data, e.g. on an `ImageCache` that is
    still parsing; it is a daemon thread, and it exits once that data arrives.
    """
    def __init__(self, iterable, fn=None, depth=2, poll=0.1):
        if depth < 1:
            raise ValueError(f"prefetch depth must be positive, got {depth}")
        self._queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._poll = poll
        self._finished = False
        # The thread gets the queue and the event, not `self`, so that
        # dropping the last reference to this object closes it.
        self._thread = threading.Thread(
            target=self._work,
            args=(iterable, fn, self._queue, self._stop, poll),
            name="abismal-prefetch", daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _work(iterable, fn, q, stop, poll):
        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=poll)
                    return True
                except queue.Full:
                    continue
            return False

        it = iter(iterable)
        try:
            for item in it:
                if stop.is_set():
                    return
                if fn is not None:
                    item = fn(item)
                if not put((False, item)):
                    return
        except BaseException as err:
            put((True, err))
            return
        finally:
            # Close upstream generators here, in the thread that ran them.
            close = getattr(it, 'close', None)
            if close is not None:
                close()
        put((True, _END))

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        while True:
            try:
                is_control, item = self._queue.get(timeout=self._poll)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    self._finished = True
                    raise StopIteration
        if not is_control:
            return item
        self._finished = True
        if item is _END:
            raise StopIteration
        raise item

    def close(self):
        """Stop the background thread and drop anything it has queued."""
        self._finished = True
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


# `BatchStream.iterate` takes a `prefetch` argument, which hides the class.
_prefetch = prefetch


class BatchStream:
    """Batches of one split of an `ImageCache`, as the trainer consumes them.

    Iterating gives a fresh iterator of `Batch`es each time: the split's images
    (the first pass waits on parsing as needed), reshuffled through a
    `shuffle_buffer_size` reservoir and collated `batch_size` images at a time.
    With `repeat=True` the iterator never ends, and one iterator should be kept
    across epochs, taking `steps_per_epoch` batches from it each time; that is
    how Keras consumed the repeated tf.data pipeline.

    `iterate(transfer=fn, prefetch=depth)` is the same with `fn` (say, a copy
    to the GPU) applied to every batch, and the whole chain run `depth` batches
    ahead in a background thread. Plain iteration does no prefetching.

    `seed` seeds the shuffle; None, the default, gives a different order on
    every run, as tf.data's unseeded shuffle did. Either way each new iterator
    draws a new order.
    """
    def __init__(self, cache, split='train', batch_size=100, shuffle_buffer_size=0, repeat=False, seed=None,
                 reflections_per_image=None, sample_with_replacement=True):
        self.reflections_per_image = reflections_per_image
        self.sample_with_replacement = sample_with_replacement
        self.cache = cache
        self.split = split
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.repeat = repeat
        self.rng = np.random.default_rng(seed)

    def images(self, repeat=None):
        """The split's images, unshuffled and unbatched."""
        if repeat is None:
            repeat = self.repeat
        return self.cache.iterate(self.split, repeat=repeat)

    def batches(self):
        images = reservoir_shuffle(self.images(), self.shuffle_buffer_size, self.rng)
        k = self.reflections_per_image
        if k is None:
            return batched(images, self.batch_size)
        # Fixed-shape batches: every image resampled to k reflections, and no
        # short final batch.
        images = (sample_reflections(im, k, self.rng, self.sample_with_replacement) for im in images)
        return batched(images, self.batch_size, drop_remainder=True)

    def __iter__(self):
        return self.batches()

    @property
    def metadata_length(self):
        """Width of the metadata, known once the first image is parsed."""
        return self.cache.first_image()[4].shape[-1]

    def iterate(self, transfer=None, prefetch=2):
        """Batches with `transfer` applied, prefetched `prefetch` deep.

        `prefetch=0` runs everything in the caller's thread instead.
        """
        if prefetch and prefetch > 0:
            return _prefetch(self.batches(), fn=transfer, depth=prefetch)
        if transfer is None:
            return self.batches()
        return map(transfer, self.batches())

    @property
    def done(self):
        """Whether all images have been parsed."""
        return self.cache.done
