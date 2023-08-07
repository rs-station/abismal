from abismal.distributions import FoldedNormal, Rice
import tensorflow as tf
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import dtype_util
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
import numpy as np
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.bijectors import exp as exp_bijector

class RiceWoolfson(distribution.AutoCompositeTensorDistribution):
    def __init__(self,
		   loc,
		   scale,
           centric,
		   validate_args=False,
		   allow_nan_stats=True,
		   name='Rice'):

        parameters = dict(locals())
        dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
        with tf.name_scope(name) as name:
            self._loc = tensor_util.convert_nonref_to_tensor(loc)
            self._scale = tensor_util.convert_nonref_to_tensor(scale)
            self._centric = tensor_util.convert_nonref_to_tensor(centric, dtype='bool')
        super(RiceWoolfson, self).__init__(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            parameters=parameters,
            name=name)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        pp = dict(
            loc=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: exp_bijector.Exp()
                )
            ),
            centric=parameter_properties.ParameterProperties(),
        )
        return pp
        # pylint: enable=g-long-lambda

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def centric(self):
        return self._centric

    @property
    def _centric_distribution(self):
        return FoldedNormal(self.loc, self.scale)

    @property
    def _acentric_distribution(self):
        return Rice(self.loc, self.scale)

    def _log_prob(self, X, **kwargs):
        return tf.where(
            self._centric,
            self._centric_distribution.log_prob(X),
            self._acentric_distribution.log_prob(X),
        )

    def _sample_n(self, n, seed=None):
        return tf.where(
            self._centric,
            self._centric_distribution.sample(n, seed=seed),
            self._acentric_distribution.sample(n, seed=seed),
        )

    def _mean(self):
        return tf.where(
            self._centric,
            self._centric_distribution.mean(),
            self._acentric_distribution.mean(),
        )

    def _variance(self):
        return tf.where(
            self._centric,
            self._centric_distribution.variance(),
            self._acentric_distribution.variance(),
        )

    def _stddev(self):
        return tf.where(
            self._centric,
            self._centric_distribution.stddev(),
            self._acentric_distribution.stddev(),
        )

    def sample_intensities(self, sample_shape=(), seed=None, name='sample', **kwargs):
        return tf.where(
            self.centric,
            self._centric_distribution.sample_square(sample_shape=sample_shape, seed=seed, name=name, **kwargs),
            self._acentric_distribution.sample_square(sample_shape=sample_shape, seed=seed, name=name, **kwargs),
        )

@kullback_leibler.RegisterKL(RiceWoolfson, RiceWoolfson)
def _kl_rice_rice_woolfson(q, p, name=None):
    return tf.where(
        q.centric,
        q._centric_distribution.kl_divergence(p._centric_distribution),
        q._acentric_distribution.kl_divergence(p._acentric_distribution),
    )



if __name__=="__main__":
    n = 100
    loc,scale = np.random.random((2, n)).astype('float32')
    centric = np.random.choice([False, True], size=n)
    q = RiceWoolfson(loc, scale, centric)
    p = RiceWoolfson(0., 1., centric)
    x = np.linspace(-10., 10., 1000)
    u = q.mean()
    s = q.stddev()
    v = q.variance()

    q.log_prob(x[:,None])
    q.prob(x[:,None])
    q.sample(n)
    q.sample_intensities(n)
    q.kl_divergence(p)

    from IPython import embed
    embed(colors='linux')

