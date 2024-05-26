import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd

def weighted_pearsonr(x, y, w=None, axis=-1, keepdims=False, eps=1e-12):
    """
    Calculate a [weighted Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient).

    Note
    ----
    x, y, and w may have arbitrarily shaped leading dimensions. The correlation coefficient will always be computed pairwise along the last axis.

    Parameters
    ----------
    x : np.array(float)
        An array of observations.
    y : np.array(float)
        An array of observations the same shape as x.
    w : np.array(float)
        An array of weights the same shape as x. These needn't be normalized.

    Returns
    -------
    r : float
        The Pearson correlation coefficient along the last dimension. This has shape {x,y,w}.shape[:-1].
    """
    if w is None:
        w = tf.ones_like(x)

    z = tf.math.reciprocal(tf.reduce_sum(w, axis=axis, keepdims=True))
    mx = tf.reduce_sum(z * (w * x), axis=axis, keepdims=True)
    my = tf.reduce_sum(z * (w * y), axis=axis, keepdims=True)

    dx = x - mx
    dy = y - my

    cxy = z * tf.reduce_sum(w * dx * dy, axis=axis, keepdims=True)
    cx = z * tf.reduce_sum(w * dx * dx, axis=axis, keepdims=True)
    cy = z * tf.reduce_sum(w * dy * dy, axis=axis, keepdims=True)

    r = cxy / tf.sqrt(cx * cy + eps)
    return r

class LocationScale(tfk.layers.Layer):
    def _likelihood(self, iobs, sigiobs):
        raise NotImplementedError(
            "Derived classes must implement _likelihood(iobs, sigiobs)->tfd.Distribution"
        )

    def register_metrics(self, ipred, iobs, sigiobs):
        likelihood = self._likelihood(iobs, sigiobs)

        # This is the mean ipred across the posterior mc samples
        iobs = iobs * tf.ones_like(ipred)
        sigiobs = sigiobs * tf.ones_like(ipred)
        w = tf.math.reciprocal(tf.square(sigiobs))

        cc = weighted_pearsonr(iobs, ipred, w, axis=(-2, -1))
        cc = tf.squeeze(cc)
        self.add_metric(cc, name='wCCpred')

        cc = weighted_pearsonr(iobs, ipred, axis=(-2, -1))
        cc = tf.squeeze(cc)
        self.add_metric(cc, name='CCpred')

        cdf = likelihood.cdf(ipred)
        f = tf.reduce_mean(
            tf.where(
                (cdf >= 0.025)&(cdf <= 0.975),
                1.,
                0.,
            )
        )
        self.add_metric(f, "C95")

        resid = ipred - iobs
        r2 = tf.square(resid)

        mse = tf.reduce_mean(r2)
        self.add_metric(mse, name='MSE')

        mae = tf.reduce_mean(tf.abs(resid))
        self.add_metric(mse, name='MAE')

        wmse = tf.reduce_sum(w * r2) / tf.reduce_sum(w * tf.ones_like(r2))
        self.add_metric(wmse, name='WMSE')

        z = tf.abs(resid / sigiobs) 
        z1 = tf.reduce_mean(tf.where(z > 1., 0., 1.))
        z2 = tf.reduce_mean(tf.where(z > 2., 0., 1.))
        z3 = tf.reduce_mean(tf.where(z > 3., 0., 1.))
        self.add_metric(z1, name='Z1_Frac')
        self.add_metric(z2, name='Z2_Frac')
        self.add_metric(z3, name='Z3_Frac')

    def call(self, ipred, iobs, sigiobs):
        likelihood = self._likelihood(iobs, sigiobs)
        ll = likelihood.log_prob(ipred)
        return ll

