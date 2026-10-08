import numpy as np
import torch
from abismal import serialization


@serialization.register
class FourierFeatures(torch.nn.Module, serialization.Serializable):
    r"""Random Fourier features with a declared length scale.

    Maps ``x`` to ``[cos(Bx), sin(Bx)] / sqrt(num_frequencies)`` for a fixed,
    randomly drawn ``B``. Drawing ``B ~ N(0, 1 / length_scale**2)`` makes the
    inner product of two encodings an unbiased estimate of a Gaussian kernel
    (Rahimi & Recht 2007):

    .. math::
        z(x) \cdot z(y) \;\approx\; \exp\!\left(-\frac{\|x-y\|^2}{2\ell^2}\right)

    so the encoding itself carries no structure finer than ``length_scale``,
    and unlike input noise it is a property of the layer rather than of the
    training loop -- it is in force at evaluation time too.

    How far that constrains the model depends on what reads the features. A
    *linear* readout inherits the bound: fitting a target eight times beyond
    the band at ``length_scale=1`` reaches only r = 0.196. A nonlinear one does
    not -- a 3-layer MLP on the same features reaches r = 1.000, because
    products of the cos/sin terms generate sum frequencies. So treat the length
    scale as a smoothness prior on the input representation, not a bound on the
    function a deep network can express through it.

    ``B`` is a non-trainable buffer. It is drawn once at build time and then
    saved and restored with the model -- a redrawn ``B`` would be a different
    function, so a checkpoint that lost it would be meaningless. Leaving it
    fixed is also what keeps the length scale honest; see ``trainable_B``.

    The raw input is deliberately *not* concatenated onto the output. Passing
    the coordinate through alongside its encoding would let the network
    represent arbitrarily fine variation and give the length scale away.

    Notes
    -----
    Acts on the last axis, so it can stand in for a ``Linear`` on the flat,
    per-reflection batches used throughout abismal.

    The length scale is in units of the input's own scale. Metadata reaching
    the scale model has already been standardized per column, so a scalar
    ``length_scale`` there is a fraction of a column's standard deviation, and
    transfers across datasets without rescaling.
    """

    def __init__(
        self,
        num_frequencies,
        length_scale=1.0,
        seed=None,
        trainable_B=False,
        in_features=None,
    ):
        """
        Parameters
        ----------
        num_frequencies : int
            Number of frequency vectors to draw. The output has
            ``2 * num_frequencies`` channels, one cos and one sin per vector.
            The kernel approximation error falls off as
            ``1 / sqrt(num_frequencies)``.
        length_scale : float or array (optional)
            The scale below which the encoded function may not vary. A scalar
            applies to every input column; an array of the input width sets one
            per column, for metadata whose columns are not equally smooth.
            Defaults to 1.
        seed : int (optional)
            Seed for drawing ``B``. Supply one if you want two runs to share an
            encoding; the drawn values are saved with the model either way.
        trainable_B : bool (optional)
            Whether ``B`` is trainable. False by default, and worth leaving
            that way: training ``B`` lets the network recover whatever
            bandwidth minimizes the loss, which is the behaviour the length
            scale exists to prevent. Set it only if you want the features and
            have given up the guarantee.
        in_features : int (optional)
            The input width. If omitted, ``B`` is drawn on the first call.
        """
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.length_scale = length_scale
        self.seed = seed
        self.trainable_B = trainable_B
        self.in_features = None
        self.register_buffer("B", None)
        if in_features is not None:
            self.build(in_features)

    def build(self, d):
        d = int(d)
        self.in_features = d

        ell = np.broadcast_to(
            np.asarray(self.length_scale, dtype="float32"), (d,)
        ).astype("float32")
        if np.any(ell <= 0):
            raise ValueError(
                f"length_scale must be positive, got {self.length_scale}"
            )

        # B ~ N(0, 1 / ell**2). The row scaling is what makes the induced
        # kernel exp(-||x-y||**2 / (2 ell**2)); a per-column ell simply gives
        # each input direction its own bandwidth.
        rng = np.random.default_rng(self.seed)
        B = rng.standard_normal((d, self.num_frequencies)).astype("float32")
        B = torch.as_tensor(B / ell[:, None])

        if self.trainable_B:
            del self.B
            self.B = torch.nn.Parameter(B)
        else:
            self.B = B

    def forward(self, data):
        if self.B is None:
            self.build(data.shape[-1])
        data = torch.as_tensor(data)
        projected = data.to(self.B.dtype) @ self.B
        # 1/sqrt(num_frequencies), not sqrt(2/num_frequencies): the cos and sin
        # pair already supplies the factor of two, so this normalization is
        # what makes z(x).z(y) an unbiased estimate of the kernel rather than
        # twice it.
        norm = self.num_frequencies ** -0.5
        return torch.cat(
            (torch.cos(projected), torch.sin(projected)), dim=-1
        ) * norm

    def compute_output_shape(self, shape):
        return tuple(shape[:-1]) + (2 * self.num_frequencies,)

    def get_config(self):
        length_scale = self.length_scale
        if isinstance(length_scale, np.ndarray):
            length_scale = length_scale.tolist()
        return {
            "num_frequencies": self.num_frequencies,
            "length_scale": length_scale,
            "seed": self.seed,
            "trainable_B": self.trainable_B,
            "in_features": self.in_features,
        }
