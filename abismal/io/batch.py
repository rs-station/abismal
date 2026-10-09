from typing import Any, NamedTuple

import numpy as np


class Batch(NamedTuple):
    """A ragged batch of images, flattened to one row per reflection.

    The first seven fields follow the per-image record the loaders yield
    (see `DataLoader.data_labels`). `image_id` says which of the `num_images`
    images each reflection came from and is sorted, because images are
    concatenated in order. Row splits are not stored; recover them with
    `np.bincount(image_id, minlength=num_images)` if needed.

    The class is framework free. The trainer moves a batch to a device with
    `batch.map(lambda a: torch.as_tensor(a).to(device, non_blocking=True))`.
    """
    asu_id: Any      # [N,1] int32
    hkl: Any         # [N,3] int32
    resolution: Any  # [N,1] float32
    wavelength: Any  # [N,1] float32
    metadata: Any    # [N,m] float32
    iobs: Any        # [N,1] float32
    sigiobs: Any     # [N,1] float32
    image_id: Any    # [N]   int64
    num_images: int  # B
    half: Any = None  # [N] int64, or None. Which random half (0 or 1) each
                      # reflection's image belongs to; see abismal.io.pipeline.image_half

    def map(self, fn):
        """Apply `fn` to every array field, keeping `num_images`."""
        half = None if self.half is None else fn(self.half)
        return Batch(*(fn(a) for a in self[:8]), self.num_images, half)

    @property
    def inputs(self):
        return tuple(self[:7])

    @classmethod
    def collate(cls, images, halves=None):
        """Concatenate a list of per-image 7-tuples into one Batch."""
        images = list(images)
        if len(images) == 0:
            raise ValueError("Cannot collate an empty list of images")
        fields = [np.concatenate(f, axis=0) for f in zip(*images)]
        lengths = [len(image[0]) for image in images]
        image_id = np.repeat(np.arange(len(images), dtype=np.int64), lengths)
        half = None
        if halves is not None:
            half = np.repeat(np.asarray(halves, dtype=np.int64), lengths)
        return cls(*fields, image_id, len(images), half)
