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
    # Optional per-row weights, [N] float32; None means all ones. Rows with
    # context_weight 0 take no part in what an image's reflections share (the
    # normalization, the pooled image representation, the choice of reindexing
    # operator). Rows with loss_weight 0 do not count toward the objective or
    # the metrics. See abismal.io.pipeline.holdout_mask for a use of both.
    context_weight: Any = None
    loss_weight: Any = None

    def map(self, fn):
        """Apply `fn` to every array field, keeping `num_images`."""
        weights = [None if w is None else fn(w) for w in (self.context_weight, self.loss_weight)]
        return Batch(*(fn(a) for a in self[:8]), self.num_images, *weights)

    @property
    def inputs(self):
        return tuple(self[:7])

    @classmethod
    def collate(cls, images, context_weight=None, loss_weight=None):
        """
        Concatenate a list of per-image 7-tuples into one Batch, with optional
        per-image lists of row weights.
        """
        images = list(images)
        if len(images) == 0:
            raise ValueError("Cannot collate an empty list of images")
        fields = [np.concatenate(f, axis=0) for f in zip(*images)]
        lengths = [len(image[0]) for image in images]
        image_id = np.repeat(np.arange(len(images), dtype=np.int64), lengths)
        weights = [
            None if w is None else np.concatenate(w).astype(np.float32)
            for w in (context_weight, loss_weight)
        ]
        return cls(*fields, image_id, len(images), *weights)
