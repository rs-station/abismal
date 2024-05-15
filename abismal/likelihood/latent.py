import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import util as tfu
from abismal.distributions.folded_normal import FoldedNormal

def make_fn_posterior(loc, scale, epsilon=1e-12):
    loc = tf.convert_to_tensor(loc)
    scale = tf.convert_to_tensor(scale)

    loc = tf.Variable(loc)
    scale = tfu.TransformedVariable(
        scale,
        tfb.Chain([
            tfb.Shift(epsilon),
            tfb.Exp(),
        ]),
    )
    return FoldedNormal(loc, scale)

class AdaptiveLikelihoodBase(tfk.layers.Layer):
    """
    A Bayesian additive error model inspired by EV11 wherein
    sigi = a * sqrt(sigiobs**2. + b * iobs + c * iobs**2.)
    a ~ HalfNormal(1.)
    """
    def __init__(self, prior=None, kl_weight=1., epsilon=1e-6, scale_factor=1e-2):
        super().__init__()
        self.kl_weight = kl_weight
        self.prior = prior
        if self.prior is None:
            self.prior = tfd.HalfNormal(1.)

        self.qa = make_fn_posterior(
            1.,
            scale_factor * self.prior.stddev(),
            epsilon=epsilon,
        )
        self.qb = make_fn_posterior(
            epsilon,
            scale_factor * self.prior.stddev(),
            epsilon=epsilon,
        )
        self.qc = make_fn_posterior(
            epsilon,
            scale_factor * self.prior.stddev(),
            epsilon=epsilon,
        )

    def _distribution(self, iobs, sigiobs):
        raise NotImplementedError('Extensions of this class must implement _distribution')

    def call(self, ipred, iobs, sigiobs):
        mc_samples = tf.shape(ipred)[-1]
        a = self.qa.sample(mc_samples)
        b = self.qb.sample(mc_samples)
        c = self.qc.sample(mc_samples)

        #var = sigiobs * sigiobs + b[...,:] * tf.abs(iobs) + c[...,:] * tf.square(iobs)
        var = sigiobs * sigiobs + b[...,:] * tf.abs(ipred) + c[...,:] * tf.square(ipred)
        sigi = a[...,:] * tf.sqrt(var)
        ll = self._distribution(iobs, sigi).log_prob(ipred)

        q = (
            self.qa.log_prob(a) +
            self.qb.log_prob(b) +
            self.qc.log_prob(c)
        )
        p = (
            self.prior.log_prob(a) +
            self.prior.log_prob(b) +
            self.prior.log_prob(c)
        )
        kl_div = tf.reduce_mean(q - p)
        self.add_metric(kl_div, "KL_LL")
        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(self.qa.mean(), 'sdfac')
        self.add_metric(self.qb.mean(), 'sdadd')
        self.add_metric(self.qc.mean(), 'sdb')

        return ll

class AdaptiveNormalLikelihood(AdaptiveLikelihoodBase):
    def _distribution(self, iobs, sigiobs):
        return tfd.Normal(iobs, sigiobs)

class AdaptiveTLikelihood(AdaptiveLikelihoodBase):
    def __init__(self, dof, prior=None, kl_weight=1., epsilon=1e-6, scale_factor=1e-2):
        super().__init__(prior, kl_weight, epsilon, scale_factor)
        self.dof = dof

    def _distribution(self, iobs, sigiobs):
        return tfd.StudentT(self.dof, iobs, sigiobs)

