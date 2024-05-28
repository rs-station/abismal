import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk

class PosteriorBase(tfk.models.Model):
    def __init__(self, rac, epsilon=1e-12, kl_weight=1., **kwargs):
        """
        rac : ReciprocalASUCollection
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.kl_weight = kl_weight
        self.rac = rac
        self.seen = self.add_weight(
            shape=self.rac.asu_size,
            initializer='zeros',
            dtype='bool',
            trainable=False,
            name="hkl_tracker",
        )

    def register_seen(self, asu_id, hkl):
        unique,_ = tf.unique(tf.reshape(self.rac._miller_ids(asu_id, hkl), [-1]))
        unique = unique[unique!=-1]
        seen_batch = tf.scatter_nd(
            unique[:,None], 
            tf.ones_like(unique, dtype='bool'), 
            shape=[self.rac.asu_size]
        )
        self.seen.assign(self.seen | seen_batch)

    def flat_distribution(self):
        raise NotImplementedError("Subclasses must implement a flat_distribution method")

    def flat_prior(self):
        raise NotImplementedError("Subclasses must implement a flat_prior method")

    def to_datasets(self, seen=True):
        """
        Parameters
        ----------
        seen : bool (optional)
            Only include reflections seen during training. Defaults to True. 
        """
        raise NotImplementedError("Subclasses must implement a to_datasets method")

    def mean(self, asu_id, hkl):
        q = self.flat_distribution
        mean = self.rac.gather(q.mean(), asu_id, hkl)
        return mean

    def stddev(self, asu_id, hkl):
        q = self.flat_distribution
        stddev = self.rac.gather(q.stddev(), asu_id, hkl)
        return stddev

    def register_kl(self, samples=None, asu_id=None, hkl=None, training=None):
        """
        this method will always try to use an analytical kl divergence. failing
        that, the samples will be used
        """
        if training:
            q,p = self.flat_distribution(), self.flat_prior()
            try:
                q.kl_divergence(p)
            except NotImplementedError:
                kl_div = q.log_prob(samples) - p.log_prob(samples)
            kl_div = tf.reduce_mean(kl_div)
            self.add_metric(kl_div, name='KL')
            self.add_loss(self.kl_weight * kl_div)

    def get_flat_fsigf(self):
        msg = """
        Subclasses must implement get_flat_isigi and/or get_flat_fsigf
        """
        raise NotImplementedError(msg)

    def get_flat_isigi(self):
        msg = """
        Subclasses must implement get_flat_isigi and/or get_flat_fsigf
        """
        raise NotImplementedError(msg)

    def to_datasets(self, seen=True):
        h,k,l = self.rac.Hunique.numpy().T
        q = self.flat_distribution()
        data = {

            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
        }

        try:
            I,SIGI = self.get_flat_isigi()
            has_isigi = True
            data.update({
                'I' : rs.DataSeries(I, dtype='J'),
                'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
            })
        except NotImplementedError:
            has_isigi = False

        try:
            F,SIGF = self.get_flat_fsigf()
            has_fsigf = True
            data.update({
                'F' : rs.DataSeries(F, dtype='F'),
                'SIGF' : rs.DataSeries(SIGF, dtype='Q'),
            })
        except NotImplementedError:
            has_fsigf = False

        if not (has_fsigf or has_isigi):
            raise NotImplementedError


        asu_id = self.rac.asu_id
        for i,rasu in enumerate(self.rac):
            ds = rs.DataSet(
                data,
                merged=True,
                cell=rasu.cell,
                spacegroup=rasu.spacegroup,
            )
            idx = self.rac.asu_id.numpy() == i
            if seen:
                idx = idx & self.seen.numpy()

            out = ds[idx]
            out = out.set_index(['H', 'K', 'L'])
            if rasu.anomalous:
                out = out.unstack_anomalous()
                keys = []
                if has_fsigf:
                    keys += [
                        'F(+)',
                        'SIGF(+)',
                        'F(-)',
                        'SIGF(-)',
                    ]
                if has_isigi:
                    keys += [
                        'I(+)',
                        'SIGI(+)',
                        'I(-)',
                        'SIGI(-)',
                    ]
                out = out[keys]
            yield out

class StructureFactorPosteriorBase(PosteriorBase):
    parameterization = 'structure_factor'

    def get_flat_fsigf(self):
        q = self.flat_distribution()
        F = q.mean()      
        SIGF = q.stddev()
        return F, SIGF

    def get_flat_isigi(self):
        """
        This method is approximate, but the intensity is exact. It is calculated 
        based on the definition of variance. 
        The uncertainties are based on 1st order uncertainty propagation and are
        therefore less accurate. 
        Subclasses may choose to implement analytical expressions where available.
        """
        F,SIGF = self.get_flat_fsigf()
        I = F*F + SIGF*SIGF
        SIGI = np.abs(2*F*SIGF)
        return I,SIGI

class IntensityPosteriorBase(PosteriorBase):
    parameterization = 'intensity'

    def get_flat_isigi(self):
        q = self.flat_distribution()
        I = q.mean()      
        SIGI = q.stddev()
        return I, SIGI

    def get_flat_fsigf(self):
        msg = """
        Subclasses may implement get_flat_fsigf if the parameterization allows
        if the moments of the square root of the intensity distribution are known.
        """
        raise NotImplementedError(msg)

