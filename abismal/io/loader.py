import numpy as np


class DataLoader():
    """Base class for the reflection file readers.

    `get_dataset` returns an iterable of per-image records. Each record is a
    tuple of seven numpy arrays in the order of `data_labels`, one row per
    reflection: asu_id int32 [n,1], hkl int32 [n,3], resolution float32 [n,1],
    wavelength float32 [n,1], metadata float32 [n,m], I float32 [n,1] and
    SigI float32 [n,1].
    """
    data_labels = (
        "ASU",
        "HKL",
        "Resolution",
        "Wavelength",
        "Metadata",
        "I",
        "SigI",
    ) #To support laue add wavelength and resolution here
    def __init__(self, metadata_length):
        self.metadata_length = metadata_length


def make_record(asu_id, hkl, resolution, wavelength, metadata, iobs, sigiobs):
    """Build one image record with the canonical shapes and dtypes.

    `asu_id` and `wavelength` may be scalars, in which case they are
    broadcast over the reflections. The 1-D per-reflection arrays gain their
    trailing unit axis here so the loaders need not agree on it.
    """
    hkl = np.asarray(hkl, dtype=np.int32).reshape((-1, 3))
    n = len(hkl)

    def column(x, dtype):
        x = np.asarray(x, dtype=dtype)
        if x.ndim == 0:
            return np.full((n, 1), x, dtype=dtype)
        return x.reshape((n, 1))

    metadata = np.asarray(metadata, dtype=np.float32)
    if metadata.ndim != 2:
        metadata = metadata.reshape((n, -1))
    return (
        column(asu_id, np.int32),
        hkl,
        column(resolution, np.float32),
        column(wavelength, np.float32),
        metadata,
        column(iobs, np.float32),
        column(sigiobs, np.float32),
    )


def split_by_image(image_ids, *arrays):
    """Yield the rows of each array belonging to each image, in image order.

    `image_ids` must be sorted. Yields one tuple of slices per distinct id.
    """
    image_ids = np.asarray(image_ids)
    if len(image_ids) == 0:
        return
    bounds = np.flatnonzero(np.diff(image_ids)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(image_ids)]))
    for start, stop in zip(starts, stops):
        yield tuple(a[start:stop] for a in arrays)
