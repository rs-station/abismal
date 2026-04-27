import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.bijectors import softplus as softplus_bijector


class Nakagami(tfd.TransformedDistribution):
    """The Nakagami distribution.

    If ``X ~ Gamma(concentration=m, rate=m/Omega)`` then
    ``Y = sqrt(X) ~ Nakagami(m, Omega)``. This implementation wraps
    ``tfd.Gamma`` with ``tfb.Invert(tfb.Square())`` so reparameterized
    sampling comes directly from the underlying Gamma.

    The density is

        f(y; m, Omega) = 2 m^m / (Gamma(m) Omega^m) y^{2m - 1}
                         exp(-m y^2 / Omega),    y >= 0, m >= 1/2, Omega > 0.

    Special cases relevant to Wilson's distributions of X-ray structure
    factor amplitudes:

      * ``Nakagami(m=0.5, Omega=sigma**2)`` is ``HalfNormal(scale=sigma)``
        (the centric Wilson prior).
      * ``Nakagami(m=1.0, Omega=sigma**2)`` is
        ``Weibull(concentration=2., scale=sigma)`` (the acentric Wilson prior).
    """

    def __init__(self,
                 concentration,
                 spread,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Nakagami'):
        """Construct a Nakagami distribution.

        Args:
          concentration: Floating-point `Tensor`; the shape parameter ``m``
            (also called ``nu`` in some references). Must be >= 0.5 for a
            proper density; no runtime check is enforced unless
            ``validate_args=True``.
          spread: Floating-point `Tensor`; the spread parameter ``Omega``
            equal to ``E[Y^2]``. Must be strictly positive.
          validate_args: Python `bool`, default `False`. Whether to validate
            input with asserts.
          allow_nan_stats: Python `bool`, default `True`. If `False`, raise
            an exception if a statistic (e.g. mean/mode/etc.) is undefined
            for any batch member.
          name: Python `str`, name prefixed to Ops created by this class.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype(
                [concentration, spread], dtype_hint=tf.float32)
            self._concentration = tensor_util.convert_nonref_to_tensor(
                concentration, dtype=dtype, name='concentration')
            self._spread = tensor_util.convert_nonref_to_tensor(
                spread, dtype=dtype, name='spread')

            gamma = tfd.Gamma(
                concentration=self._concentration,
                rate=self._concentration / self._spread,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
            )
            super().__init__(
                distribution=gamma,
                bijector=tfb.Invert(tfb.Square()),
                validate_args=validate_args,
                parameters=parameters,
                name=name,
            )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            concentration=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            spread=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        )

    @property
    def concentration(self):
        return self._concentration

    @property
    def spread(self):
        return self._spread

    def square(self):
        """Return the Gamma distribution of the squared random variable.

        If ``Y ~ Nakagami(m, Omega)`` then ``Y**2 ~ Gamma(m, m / Omega)``.
        """
        m = tf.convert_to_tensor(self._concentration)
        omega = tf.convert_to_tensor(self._spread)
        return tfd.Gamma(concentration=m, rate=m / omega)

    def _log_gamma_ratio(self):
        """Return ``lgamma(m + 1/2) - lgamma(m)`` computed stably."""
        m = tf.convert_to_tensor(self._concentration)
        half = tf.constant(0.5, dtype=m.dtype)
        return -tfm.log_gamma_difference(half, m)

    def _mean(self):
        m = tf.convert_to_tensor(self._concentration)
        omega = tf.convert_to_tensor(self._spread)
        return tf.math.exp(
            self._log_gamma_ratio() + 0.5 * (tf.math.log(omega) - tf.math.log(m))
        )

    def _variance(self):
        m = tf.convert_to_tensor(self._concentration)
        omega = tf.convert_to_tensor(self._spread)
        ratio_sq = tf.math.exp(2.0 * self._log_gamma_ratio())
        return omega * (1.0 - ratio_sq / m)

    def _stddev(self):
        return tf.math.sqrt(self._variance())

    def _mode(self):
        m = tf.convert_to_tensor(self._concentration)
        omega = tf.convert_to_tensor(self._spread)
        half = tf.constant(0.5, dtype=m.dtype)
        return tf.math.sqrt(tf.maximum(0.0, (m - half) * omega / m))


@kullback_leibler.RegisterKL(Nakagami, Nakagami)
def _kl_nakagami_nakagami(a, b, name=None):
    """Analytical KL divergence between two Nakagami distributions.

    KL is invariant under deterministic bijective transformations, so
    KL(Nakagami_a || Nakagami_b) = KL(Gamma_a || Gamma_b), and TFP
    already registers the latter analytically.
    """
    with tf.name_scope(name or 'kl_nakagami_nakagami'):
        return kullback_leibler.kl_divergence(a.distribution, b.distribution)
