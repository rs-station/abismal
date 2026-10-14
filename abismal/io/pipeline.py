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
  - Shuffling is by pass. A pass that starts while parsing is still under way
    follows the parser, in input order; a pass that starts once every image is
    parsed visits them in a fresh random permutation, so each image appears
    exactly once per pass. (abismal 1.x shuffled through a sliding tf.data
    buffer instead, which with a buffer larger than the data repeats images
    within a batch.)
  - Batches are B whole images (`Batch.collate`); the last one may be short.
  - `prefetch` runs the whole chain, plus an optional device transfer, in a
    background thread.
"""
import queue
import threading
import zlib
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

    def images(self, split):
        """A snapshot of the images of `split` parsed so far, in input order."""
        with self._cond:
            return list(self._images[split])

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


def holdout_mask(image, fraction, seed=0):
    """
    Which of an image's reflections to hold out for validation.

    A fixed fraction, drawn at random but seeded from the image's own miller
    indices, so the same reflections are held out on every pass without the
    image having to carry an id. At least one reflection always stays in
    training, so every image keeps a representation for the scale model.
    """
    hkl = np.ascontiguousarray(image[1])
    rng = np.random.default_rng([zlib.crc32(hkl.tobytes()), seed])
    mask = rng.random(len(hkl)) < fraction
    if mask.all():
        mask[rng.integers(len(mask))] = False
    return mask


def batched(iterable, batch_size):
    """Collate consecutive images into `Batch`es of `batch_size` images.

    The last batch holds the remainder, like tf.data's `ragged_batch` with
    `drop_remainder=False`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    it = iter(iterable)
    while True:
        images = list(islice(it, batch_size))
        if not images:
            return
        yield Batch.collate(images)


def _drop_held_out(image, fraction):
    keep = ~holdout_mask(image, fraction)
    return tuple(a[keep] for a in image)


def _batched_held_out(iterable, batch_size, fraction):
    """
    Batches of whole images in which only the held-out reflections count:
    they get loss weight 1 and no part in the image's shared context, and the
    rest of each image the reverse.
    """
    it = iter(iterable)
    while True:
        images = list(islice(it, batch_size))
        if not images:
            return
        held = [holdout_mask(im, fraction) for im in images]
        yield Batch.collate(images, context_weight=[~h for h in held], loss_weight=held)


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

    Iterating gives a fresh iterator of `Batch`es each time: the split's images,
    collated `batch_size` images at a time. With `shuffle=True` each pass that
    starts after parsing is complete visits the images in a fresh random
    permutation; a pass that starts while parsing is under way follows the
    parser in input order (and waits on it as needed).
    With `repeat=True` the iterator never ends, and one iterator should be kept
    across epochs, taking `steps_per_epoch` batches from it each time; that is
    how Keras consumed the repeated tf.data pipeline.

    `iterate(transfer=fn, prefetch=depth)` is the same with `fn` (say, a copy
    to the GPU) applied to every batch, and the whole chain run `depth` batches
    ahead in a background thread. Plain iteration does no prefetching.

    `seed` seeds the shuffle; None, the default, gives a different order on
    every run. Either way every pass draws a new permutation.
    """
    def __init__(self, cache, split='train', batch_size=100, shuffle=False, repeat=False, seed=None,
                 holdout_fraction=None, holdout_role=None):
        """
        `holdout_fraction` and `holdout_role` hold out a fraction of every
        image's reflections (see `holdout_mask`). With the role 'train' those
        reflections are dropped; with 'validate' every reflection is kept and
        only the held-out ones count (see `abismal.io.Batch` for the weights).
        """
        if holdout_role not in (None, 'train', 'validate'):
            raise ValueError(f"holdout_role must be 'train' or 'validate', not {holdout_role!r}")
        self.holdout_fraction = holdout_fraction
        self.holdout_role = holdout_role
        self.cache = cache
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.rng = np.random.default_rng(seed)

    def _one_pass(self):
        if self.shuffle and self.cache.done:
            self.cache.wait()  # re-raises a parsing error, if there was one
            images = self.cache.images(self.split)
            for i in self.rng.permutation(len(images)):
                yield images[i]
        else:
            yield from self.cache.iterate(self.split, repeat=False)

    def images(self, repeat=None):
        """The split's images, unbatched, pass after pass with `repeat=True`.

        A pass is in input order, or with `shuffle=True` and every image parsed,
        a fresh random permutation.
        """
        if repeat is None:
            repeat = self.repeat
        while True:
            n = 0
            for image in self._one_pass():
                n += 1
                yield image
            if not repeat or n == 0:
                return

    def batches(self):
        images = self.images()
        if self.holdout_role == 'train':
            images = (_drop_held_out(im, self.holdout_fraction) for im in images)
        elif self.holdout_role == 'validate':
            return _batched_held_out(images, self.batch_size, self.holdout_fraction)
        return batched(images, self.batch_size)

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
