import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import util as tfu
from abismal.distributions.folded_normal import FoldedNormal

class AdditiveNormalLikelihood(tfk.layers.Layer):
    """
    A Bayesian additive error model inspired by EV11 wherein
    sigi = sdfac * sqrt(sigiobs**2. + sdb * iobs + (sdadd*iobs)**2.)
    a ~ HalfNormal(1.)
    """
    def __init__(self, prior=None, kl_weight=1., epsilon=1e-12):
        super().__init__()
        self.kl_weight = kl_weight
        self.prior = prior
        if self.prior is None:
            self.prior = tfd.HalfNormal(1.)
        loc = tf.Variable(self.prior.mean())
        scale = tfu.TransformedVariable(
            self.prior.stddev(),
            tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ]),
        )
        self.surrogate_posterior_sdfac = FoldedNormal(
            self.prior.mean(),
            self.prior.stddev(),
        )
        self.surrogate_posterior_sdb = FoldedNormal(
            self.prior.mean(),
            self.prior.stddev(),
        )
        self.surrogate_posterior_sdadd = FoldedNormal(
            self.prior.mean(),
            self.prior.stddev(),
        )

    def call(self, ipred, iobs, sigiobs):
        mc_samples = tf.shape(ipred)[-1]
        sdfac = self.surrogate_posterior_sdfac.sample(mc_samples)
        sdb = self.surrogate_posterior_sdb.sample(mc_samples)
        sdadd = self.surrogate_posterior_sdadd.sample(mc_samples)

        var = sigiobs * sigiobs + sdb[...,:] * tf.abs(iobs) + tf.square(sdadd[...,:] * iobs)
        sigi = sdfac[...,:] * tf.sqrt(var)
        ll = tfd.Normal(iobs, sigi).log_prob(ipred)

        q = (
            self.surrogate_posterior_sdfac.log_prob(sdfac) +
            self.surrogate_posterior_sdb.log_prob(sdb) +
            self.surrogate_posterior_sdadd.log_prob(sdadd)
        )
        p = (
            self.prior.log_prob(sdfac) +
            self.prior.log_prob(sdb) +
            self.prior.log_prob(sdadd)
        )
        kl_div = tf.reduce_mean(q - p)
        self.add_metric(kl_div, "KL_LL")
        self.add_loss(self.kl_weight * kl_div)

        return ll

