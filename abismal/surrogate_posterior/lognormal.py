import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb


class LogNormalPosteriorBase(object):
    """Base class for variational posteriors parameterized by a log normal distribution.

    The log normal is the distribution of ``exp(X)`` for a normal ``X``, so its
    two parameters -- ``loc`` and ``scale`` -- are the mean and standard
    deviation of the *underlying normal*, not of the distribution itself. That
    makes ``loc`` genuinely unconstrained and needs no bijector; only ``scale``
    is kept positive, by the same shifted Exp the other posteriors use.

    Support is (0, inf), which suits both parameterizations without a shift: a
    sample can never be zero or negative, and reaching the float32 floor from a
    well-conditioned ``loc`` takes well over a hundred standard deviations.

    It is also closed under squaring and square roots, so both structure factor
    and intensity moments come out analytically in either parameterization --
    see the subclasses in ``structure_factor`` and ``intensity``.
    """

    def __init__(self, rac, loc_init=None, scale_init=None, epsilon=1e-12, **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        loc_init : array (optional)
            The *mean* the posterior should start at, in the length of the ASU.
            Note this is a moment of the distribution, not the ``loc``
            parameter; the two are related by the moment matching below. This
            follows the convention of the other posteriors in this package.
        scale_init : array (optional)
            The *standard deviation* the posterior should start at, again a
            moment rather than the ``scale`` parameter.
        epsilon : float (optional)
            Floors the moments and the positivity bijector.
        """
        super().__init__(rac, epsilon=epsilon, **kwargs)

        if loc_init is None:
            loc_init = tf.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * loc_init

        loc_init = tf.convert_to_tensor(loc_init)
        scale_init = tf.convert_to_tensor(scale_init, dtype=loc_init.dtype)

        # Moment matching. For mean m and variance s**2,
        #   scale**2 = log(1 + s**2 / m**2)
        #   loc      = log(m) - scale**2 / 2
        # Both moments are floored first: a zero mean would send loc to -inf,
        # and a zero standard deviation would send the unconstrained scale
        # variable there instead.
        mean = tf.maximum(loc_init, epsilon)
        stddev = tf.maximum(scale_init, epsilon)
        scale_sq = tf.math.log1p(tf.math.square(stddev / mean))

        # loc is the mean of a normal, so it is unconstrained and is stored
        # directly rather than through a bijector.
        self.loc = tf.Variable(
            tf.math.log(mean) - 0.5 * scale_sq,
            name="loc",
        )
        self.scale = tfu.TransformedVariable(
            tf.math.sqrt(scale_sq),
            tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ]),
        )
        self.built = True

    def _distribution(self, loc, scale):
        return tfd.LogNormal(loc, scale)

    def distribution(self, asu_id, hkl):
        loc = self.rac.gather(self.loc, asu_id, hkl)
        scale = self.rac.gather(self.scale, asu_id, hkl)
        return self._distribution(loc, scale)

    def flat_distribution(self):
        return self._distribution(self.loc, self.scale)
