import numpy as np
import torch
from abismal import tracking


class ResolutionTrend(torch.nn.Module):
    """
    A learned falloff T(d) that multiplies the surrogate posterior's intensities on their way to the likelihood and
    the output, while the posterior and its KL stay on a flat scale:

        I_obs ~ s * T(d) * F**2,    F ~ q,    KL[q(F) || flat prior]

    This is the same model as a prior whose Σ follows T(d), with the falloff in a few parameters instead of spread over
    every reflection's posterior (a non-centered parameterization), so that it can trade quickly against the scales.

    kind='wilson': log T = log k - b / d**2.  kind='spline': log T piecewise linear in 1/d**2, with `knots` knots at
    quantiles of the reflections' 1/d**2. Parameters are in units of 0.1 of log T (of b for 'wilson'), because Adam
    moves a parameter about the learning rate per step.

    anchor='lowres' pins the trend's level: T = 1 at the lowest-resolution knot ('spline') or at the lowest-resolution
    reflection ('wilson', k fixed at 1). Without it, a constant can move between the trend and the scales at no cost to
    the likelihood; with point scales and no scale prior nothing stops that drift.

    aniso='global' or 'bins' (spline only) adds an anisotropic temperature factor, log T -= s^T B s / 2 with s the
    Cartesian reciprocal vector of the reflection: one traceless B for the whole ASU, or one per knot, interpolated like
    the knot values. The isotropic part stays in the spline. B is averaged over the point group's rotations in the forward
    pass, so only the components the Laue symmetry allows ever get a gradient (5 in triclinic, 1 in hexagonal, none in
    cubic). Its parameters are in units where 1 moves log T by 10 at the edge of the data.
    """
    def __init__(self, rac, kind='wilson', knots=16, aniso='none', anchor='none'):
        super().__init__()
        if anchor not in ('none', 'lowres'):
            raise ValueError(f"anchor must be 'none' or 'lowres', not {anchor!r}")
        self.anchor = anchor
        if kind not in ('wilson', 'spline'):
            raise ValueError(f"kind must be 'wilson' or 'spline', not {kind!r}")
        if aniso not in ('none', 'global', 'bins'):
            raise ValueError(f"aniso must be 'none', 'global' or 'bins', not {aniso!r}")
        if aniso != 'none' and kind != 'spline':
            raise ValueError("an anisotropic trend needs kind='spline'")
        self.rac = rac
        self.kind = kind
        self.aniso = aniso
        q = torch.as_tensor(rac.dHKL, dtype=torch.float32).reshape(-1) ** -2
        if kind == 'wilson':
            self.register_buffer('q', q)
            self._log_k = torch.nn.Parameter(torch.tensor(0.))
            self._b = torch.nn.Parameter(torch.tensor(0.))
        else:
            edges = np.unique(np.quantile(q.numpy(), np.linspace(0, 1, knots)))
            knot = torch.as_tensor(edges, dtype=torch.float32)
            index = torch.bucketize(q, knot, right=True).clamp(1, len(knot) - 1) - 1
            frac = (q - knot[index]) / (knot[index + 1] - knot[index])
            self.register_buffer('knots', knot)
            self.register_buffer('index', index)
            self.register_buffer('frac', frac)
            self._values = torch.nn.Parameter(torch.zeros(len(knot)))
        if aniso != 'none':
            features, projection = _aniso_features(rac)
            self.register_buffer('features', features)
            self.register_buffer('projection', projection)
            self.register_buffer('asu_index', torch.as_tensor(rac.asu_id, dtype=torch.int64).reshape(-1))
            self.b_unit = 20. / float(q.max())
            n = len(self.knots) if aniso == 'bins' else 1
            self._aniso = torch.nn.Parameter(torch.zeros(len(rac.reciprocal_asus), n, 5))

    def _knot_values(self):
        """log T at the knots. anchor='lowres' holds the lowest-resolution knot at 0 (T = 1), so that the trend
        carries only the shape of the falloff and not its level, which the scales could otherwise trade against it
        without changing the likelihood."""
        v = 10. * self._values
        if self.anchor == 'lowres':
            v = torch.cat([torch.zeros_like(v[:1]), v[1:]])
        return v

    def log_trend(self):
        """log T of every reflection in the ASU, by Miller id."""
        if self.kind == 'wilson':
            if self.anchor == 'lowres':
                return -10. * self._b * (self.q - self.q.min())
            return 10. * (self._log_k - self._b * self.q)
        v = self._knot_values()
        log_t = (1. - self.frac) * v[self.index] + self.frac * v[self.index + 1]
        if self.aniso == 'none':
            return log_t
        # B of every knot as its 6 components (xx, yy, zz, xy, xz, yz), symmetrized for each reflection's ASU, then
        # contracted with that reflection's s s^T. 'global' has one knot.
        b = self.b_tensors()
        if self.aniso == 'bins':
            b = (1. - self.frac)[:, None] * b[self.asu_index, self.index] + self.frac[:, None] * b[self.asu_index, self.index + 1]
        else:
            b = b[self.asu_index, 0]
        return log_t - 0.5 * (self.features * b).sum(-1)

    def b_tensors(self):
        """The symmetrized B, in Å^2, of each ASU and knot as (xx, yy, zz, xy, xz, yz): shape (asus, knots, 6)."""
        return self.b_unit * torch.einsum('akp,aqp->akq', self._aniso, self.projection)

    def forward(self, asu_id, hkl):
        """T(d) of the given reflections, shape (reflections, 1)."""
        if self.kind == 'wilson':
            tracking.add_metric(10. * self._b.detach() * 2., 'TrendB')
            tracking.add_metric(torch.exp(10. * self._log_k.detach()), 'TrendK')
        else:
            v = self._knot_values().detach()
            tracking.add_metric(v[0] - v[-1], 'TrendRange')
        if self.aniso != 'none':
            b = self.b_tensors().detach()
            m = torch.stack([b[..., 0], b[..., 3], b[..., 4], b[..., 3], b[..., 1], b[..., 5],
                             b[..., 4], b[..., 5], b[..., 2]], -1).reshape(*b.shape[:-1], 3, 3)
            e = torch.linalg.eigvalsh(m)
            tracking.add_metric((e[..., -1] - e[..., 0]).max(), 'TrendAniso')
        return torch.exp(self.rac.gather(self.log_trend(), asu_id, hkl)).reshape(-1, 1)


def _traceless_basis():
    """Five symmetric traceless 3x3 tensors as (xx, yy, zz, xy, xz, yz) rows, orthonormal in the Frobenius norm."""
    r2, r6 = np.sqrt(2.), np.sqrt(6.)
    return np.array([
        [1 / r2, -1 / r2, 0, 0, 0, 0],
        [1 / r6, 1 / r6, -2 / r6, 0, 0, 0],
        [0, 0, 0, 1 / r2, 0, 0],
        [0, 0, 0, 0, 1 / r2, 0],
        [0, 0, 0, 0, 0, 1 / r2],
    ])


def _as_matrix(v):
    xx, yy, zz, xy, xz, yz = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _as_vector(m):
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])


def _aniso_features(rac):
    """
    For every reflection, s s^T as (sx^2, sy^2, sz^2, 2 sx sy, 2 sx sz, 2 sy sz), so that s^T B s is its dot product
    with B's (xx, yy, zz, xy, xz, yz); and for every ASU, the (6, 5) map from the five traceless parameters to B averaged
    over the point group, M B M^T with M = F^-1 R F the Cartesian action of each rotation R on reciprocal vectors s = h F.
    """
    features, projection = [], []
    basis = _traceless_basis()
    for rasu in rac.reciprocal_asus:
        frac = np.array(rasu.cell.frac.mat.tolist())
        s = rasu.Hunique @ frac
        features.append(np.stack([s[:, 0] ** 2, s[:, 1] ** 2, s[:, 2] ** 2,
                                  2 * s[:, 0] * s[:, 1], 2 * s[:, 0] * s[:, 2], 2 * s[:, 1] * s[:, 2]], -1))
        rots = {tuple(np.array(op.rot).reshape(-1)) for op in rasu.spacegroup.operations().sym_ops}
        ms = [np.linalg.solve(frac, np.array(r, dtype=float).reshape(3, 3) / 24. @ frac) for r in rots]
        projection.append(np.stack([_as_vector(np.mean([m @ _as_matrix(b) @ m.T for m in ms], 0)) for b in basis], -1))
    features = torch.as_tensor(np.concatenate(features), dtype=torch.float32)
    projection = torch.as_tensor(np.stack(projection), dtype=torch.float32)
    return features, projection
