import tensorflow as tf
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb

from abismal.distributions import Nakagami


class NakagamiPosteriorBase(object):
    """Base class for variational posteriors parameterized by a Nakagami distribution.

    The Nakagami has two free parameters: concentration ``m`` (shape) and
    spread ``Omega`` (= ``E[X**2]``). Both are kept strictly positive via a
    shifted Exp bijector; concentration is additionally lower-bounded by
    ``concentration_min`` (default ``0.5``) so that the density is proper and
    the Wilson special cases (``m = 1/2``, ``m = 1``) remain reachable.
    """

    def __init__(
        self,
        rac,
        loc_init=None,
        scale_init=None,
        epsilon=1e-12,
        concentration_min=0.5,
        **kwargs,
    ):
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.concentration_min = concentration_min

        if loc_init is None:
            loc_init = tf.ones(rac.asu_size)
        if scale_init is None:
            scale_init = 0.01 * loc_init

        loc_init = tf.convert_to_tensor(loc_init)
        scale_init = tf.convert_to_tensor(scale_init, dtype=loc_init.dtype)

        # Spread: E[X**2] = Omega exactly, so set spread = loc**2 + scale**2.
        scale_sq = scale_init * scale_init
        spread_init = loc_init * loc_init + scale_sq

        # Concentration: the large-m approximation (Var ≈ Omega / 4m) gives
        # m ≈ Omega / (4 scale**2). With the default init_scale=1.0 this
        # collapses to m ≈ 0.5 — right at the bijector floor — which puts the
        # unconstrained variable at u ≈ log(eps) ≈ -25, killing gradients.
        # Floor at 1.0 (the acentric Wilson case) to keep the unconstrained
        # variable in a well-conditioned region (u ≈ log(0.5) ≈ -0.7).
        concentration_init = spread_init / (4.0 * scale_sq)
        concentration_init = tf.maximum(concentration_init, 1.0)

        self.spread = tfu.TransformedVariable(
            spread_init,
            tfb.Chain([tfb.Shift(epsilon), tfb.Exp()]),
        )
        self.concentration = tfu.TransformedVariable(
            concentration_init,
            tfb.Chain([tfb.Shift(concentration_min + epsilon), tfb.Exp()]),
        )
        self.built = True

    def _distribution(self, concentration, spread):
        return Nakagami(concentration, spread)

    def distribution(self, asu_id, hkl):
        concentration = self.rac.gather(self.concentration, asu_id, hkl)
        spread = self.rac.gather(self.spread, asu_id, hkl)
        return self._distribution(concentration, spread)

    def flat_distribution(self):
        return self._distribution(self.concentration, self.spread)
