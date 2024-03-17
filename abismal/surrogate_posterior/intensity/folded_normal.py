import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.distributions import FoldedNormal
from abismal.surrogate_posterior import PosteriorBase
from abismal.surrogate_posterior.intensity.surrogate_posterior import WilsonPrior


class WilsonPosterior(PosteriorBase):
    parameterization = 'intensity'

    def __init__(self, rac, scale_factor=1e-1, epsilon=1e-12, kl_weight=1., **kwargs):
        super().__init__(rac, epsilon=epsilon, kl_weight=kl_weight, **kwargs)
        self.low = self.epsilon * tf.cast(self.rac.centric, dtype='float32')
        p = self.flat_prior()
        self.loc = tf.Variable(p.mean())
        #self.loc = tfu.TransformedVariable(
        #    p.mean(),
        #    tfb.Chain([
        #        tfb.Shift(epsilon + self.low), 
        #        tfb.Exp(),
        #    ]),
        #)

        #Concentration should remain above one to prevent change in curvature
        self.scale = tfu.TransformedVariable(
            scale_factor * p.stddev(),
            tfb.Chain([
                tfb.Shift(epsilon + self.low), 
                tfb.Exp(),
            ]),
        )

    def flat_prior(self):
        prior = WilsonPrior(
            self.rac.centric,
            self.rac.epsilon,
        )
        return prior

    def flat_distribution(self):
        q = FoldedNormal(
            self.loc, 
            self.scale, 
        )
        return q

    def to_datasets(self, seen=True):
        h,k,l = self.rac.Hunique.numpy().T
        q = self.flat_distribution()
        I = q.mean()      
        SIGI = q.stddev()
        asu_id = self.rac.asu_id
        for i,rasu in enumerate(self.rac):
            idx = self.rac.asu_id.numpy() == i
            if seen:
                idx = idx & self.seen.numpy()

            out = rs.DataSet({
                'H' : rs.DataSeries(h, dtype='H'),
                'K' : rs.DataSeries(k, dtype='H'),
                'L' : rs.DataSeries(l, dtype='H'),
                'I' : rs.DataSeries(I, dtype='J'),
                'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
                },
                merged=True,
                cell=rasu.cell,
                spacegroup=rasu.spacegroup,
            )[idx]

            out = out.set_index(['H', 'K', 'L'])
            if rasu.anomalous:
                out = out.unstack_anomalous()
                out = out[[
                    'I(+)',
                    'SIGI(+)',
                    'I(-)',
                    'SIGI(-)',
                ]]
            yield out

    def register_kl(self, ipred=None, asu_id=None, hkl=None, training=None):
        kl_div = 0.
        p = self.flat_prior()
        q = self.flat_distribution()
        #TODO implement low in FoldedNormal and remove this hack
        #ipred = tf.clip_by_value(ipred, self.low, np.inf) 
        ipred = ipred + self.low
        log_q = tf.reduce_mean(q.log_prob(ipred))
        log_p = tf.reduce_mean(p.log_prob(ipred))
        kl_div = log_q - log_p
        #kl_div = tf.reduce_mean(kl_div)
        self.add_metric(kl_div, name='KL')
        self.add_metric(tf.reduce_mean(log_q), name="log_q")
        self.add_metric(tf.reduce_mean(log_p), name="log_p")
        self.add_loss(self.kl_weight * kl_div)
        return kl_div

