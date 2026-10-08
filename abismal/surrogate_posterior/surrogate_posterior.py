import numpy as np
import reciprocalspaceship as rs
import torch
from abismal import serialization


def _numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class PositiveParameter(torch.nn.Module):
    """
    A trainable vector constrained to (shift, inf) as ``exp(raw) + shift``.

    This replaces TFP's ``TransformedVariable(x, Chain([Shift(shift), Exp()]))``.
    Calling the module returns the constrained values; `gather` looks the raw
    values up first and transforms only those, which keeps the per-step cost in
    the number of observed reflections rather than the size of the ASU.
    """
    def __init__(self, init, shift=0.):
        super().__init__()
        self.shift = shift
        init = torch.as_tensor(init, dtype=torch.float32)
        self.raw = torch.nn.Parameter(torch.log(init - shift))

    def transform(self, raw):
        return torch.exp(raw) + self.shift

    def forward(self):
        return self.transform(self.raw)

    def gather(self, rac, asu_id, hkl):
        return self.transform(rac.gather(self.raw, asu_id, hkl))


class PosteriorBase(torch.nn.Module, serialization.Serializable):
    anomalous_keys = [
        'F(+)',
        'SIGF(+)',
        'F(-)',
        'SIGF(-)',
        'E(+)',
        'SIGE(+)',
        'E(-)',
        'SIGE(-)',
        'I(+)',
        'SIGI(+)',
        'I(-)',
        'SIGI(-)',
    ]
    # Every trainable parameter of a posterior is indexed by miller id, so the
    # optimizer updates all of them lazily -- see abismal.optimizers.
    lazy = True

    def __init__(self, rac, epsilon=1e-12, independent=True, **kwargs):
        """
        rac : ReciprocalASUCollection
        """
        super().__init__()
        self.independent = independent
        self.epsilon = epsilon
        self.rac = rac
        self.register_buffer('seen', torch.zeros(self.rac.asu_size, dtype=torch.bool))

    def get_config(self):
        config = {
            'rac' : serialization.serialize(self.rac),
            'epsilon' : self.epsilon,
            'independent' : self.independent,
        }
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = serialization.deserialize(config['rac'])
        return cls(**config)

    @torch.no_grad()
    def register_seen(self, asu_id, hkl):
        # Invalid (-1) ids land in a sentinel slot past the end rather than being
        # masked out, which would give the update a data-dependent shape.
        ids = self.rac._miller_ids(asu_id, hkl).reshape(-1)
        ids = torch.where(ids >= 0, ids, self.rac.asu_size)
        counts = torch.zeros(self.rac.asu_size + 1, device=ids.device)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
        self.seen |= counts[:-1] > 0

    def distribution(self, asu_id, hkl):
        raise NotImplementedError("Subclasses must implement a distribution method")

    def flat_distribution(self):
        """
        This method can be overloaded to avoid unnecessary "gather" operations and increase performance.
        """
        q = self.distribution(
            self.rac.asu_id[:,None],
            self.rac.Hunique,
        )
        return q

    def mean(self, asu_id, hkl):
        q = self.flat_distribution()
        mean = self.rac.gather(q.mean, asu_id, hkl)
        return mean

    def stddev(self, asu_id, hkl):
        q = self.flat_distribution()
        stddev = self.rac.gather(q.stddev, asu_id, hkl)
        return stddev

    def compute_kl_terms(self, q, p, samples=None):
        try:
            kl_div = torch.distributions.kl_divergence(q, p)
        except NotImplementedError:
            q_z = q.log_prob(samples)
            p_z = p.log_prob(samples)
            if not self.independent:
                #it is okay to expand the dimensions to enable combining
                #a multivariate posterior and univariate prior
                if q_z.dim() < p_z.dim():
                    q_z = q_z[..., None]

            kl_div = q_z - p_z
            kl_div = kl_div.mean(0)

        return kl_div

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

    def get_flat_extras(self):
        """
        Optionally return a dictionary of additional columns.
        """
        return {}

    @torch.no_grad()
    def to_datasets(self, seen=True):
        h,k,l = _numpy(self.rac.Hunique).T
        data = {

            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
        }

        try:
            I,SIGI = map(_numpy, self.get_flat_isigi())
            has_isigi = True
            data.update({
                'I' : rs.DataSeries(I, dtype='J'),
                'SIGI' : rs.DataSeries(SIGI, dtype='Q'),
            })
        except NotImplementedError:
            has_isigi = False

        try:
            F,SIGF = map(_numpy, self.get_flat_fsigf())
            has_fsigf = True
            data.update({
                'F' : rs.DataSeries(F, dtype='F'),
                'SIGF' : rs.DataSeries(SIGF, dtype='Q'),
            })

            sqrt_mult = np.sqrt(_numpy(self.rac.epsilon))
            data.update({
                'E' : rs.DataSeries(F / sqrt_mult, dtype='E'),
                'SIGE' : rs.DataSeries(SIGF / sqrt_mult, dtype='Q'),
            })
        except NotImplementedError:
            has_fsigf = False

        if not (has_fsigf or has_isigi):
            raise NotImplementedError

        extras = self.get_flat_extras()
        data.update(extras)

        asu_id = _numpy(self.rac.asu_id)
        is_seen = _numpy(self.seen)
        for i,rasu in enumerate(self.rac):
            ds = rs.DataSet(
                data,
                merged=True,
                cell=rasu.cell,
                spacegroup=rasu.spacegroup,
            )
            idx = asu_id == i
            if seen:
                idx = idx & is_seen

            out = ds[idx]
            out = out.set_index(['H', 'K', 'L'])
            if rasu.anomalous:
                out = out.unstack_anomalous()
                keys = self.anomalous_keys.copy()
                for k in extras:
                    keys.append(f'{k}(+)')
                    keys.append(f'{k}(-)')
                out = out[[k for k in keys if k in out]]
            yield out

class StructureFactorPosteriorBase(PosteriorBase):
    parameterization = 'structure_factor'

    def get_flat_fsigf(self):
        q = self.flat_distribution()
        F = q.mean
        SIGF = q.stddev
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
        SIGI = torch.abs(2*F*SIGF)
        return I,SIGI

class IntensityPosteriorBase(PosteriorBase):
    parameterization = 'intensity'

    def get_flat_isigi(self):
        q = self.flat_distribution()
        I = q.mean
        SIGI = q.stddev
        return I, SIGI

    def get_flat_fsigf(self):
        """
        This method is approximate. It is based on simple uncertainty propagation.
        Subclasses may choose to implement analytical expressions where available.
        """
        I, SIGI = self.get_flat_isigi()
        F = torch.sqrt(I)
        SIGF = torch.abs(F * 0.5 * SIGI / I)
        return F, SIGF
