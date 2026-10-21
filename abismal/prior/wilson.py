import numpy as np
import torch
from abismal import tracking
from abismal.prior.base import PriorBase


class WilsonPriorBase(PriorBase):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, rac, sigma=1., **kwargs):
        """
        Parameters
        ----------
        rac : ReciprocalASUCollection
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. If this is an array it must be length rac.asu_size
        """
        super().__init__(rac)
        # Always a full-length buffer, even for a scalar, so that a saved model
        # restores an empirical Σ through its state_dict.
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        sigma = sigma * torch.ones(self.rac.asu_size)
        self.register_buffer('_sigma', sigma)

    @property
    def sigma(self):
        return self._sigma

    @staticmethod
    def _empirical_binned_mean(rac, dataset, bins=20, maxiter=100, isigi_cutoff=0.0):
        """Accumulate the inverse-variance-weighted mean intensity per resolution bin.

        Returns
        -------
        mean : np.ndarray
            Weighted mean intensity in each of `bins` resolution shells.
        size : np.ndarray
            Summed inverse-variance weight in each shell (an effective count).
        labels : array
            Per-reflection bin assignment for `rac.dHKL`.
        edges : array
            The `bins + 1` resolution (dHKL) bin edges.
        """
        from reciprocalspaceship.utils import bin_by_percentile
        labels,edges = bin_by_percentile(rac.dHKL.cpu().numpy(), bins=bins, ascending=False)
        labels_t = torch.as_tensor(labels, device=rac.dHKL.device)
        mean = np.zeros(bins)
        size = np.zeros(bins)
        from tqdm import tqdm
        for i,batch in tqdm(enumerate(dataset), total=maxiter):
            if i > maxiter:
                break
            asu_id = torch.as_tensor(batch.asu_id, device=labels_t.device)
            hkl = torch.as_tensor(batch.hkl, device=labels_t.device)
            idx = rac.gather(labels_t, asu_id, hkl).cpu().numpy()
            I = np.asarray(batch.iobs)[:, 0].astype('float64')
            SIGI = np.asarray(batch.sigiobs)[:, 0].astype('float64')

            if isigi_cutoff is not None:
                mask = (I / SIGI > isigi_cutoff).astype('float64')
            else:
                mask = np.ones_like(I)

            # Inverse-variance weight per observation; masked observations get zero weight.
            w = mask / np.square(SIGI)

            # `size` now accumulates summed weights rather than counts. The batched
            # Welford combine rule mean += (W_b / W) * (mean_b - mean) holds for
            # weighted means when W is the running sum of weights.
            batch_size = np.bincount(idx, w, minlength=bins)
            # Bins with zero summed weight in this batch (a resolution shell
            # absent from the batch, or fully masked by the I/sigI cutoff) get
            # batch_mean=0 and an update weight of 0, leaving the running mean
            # untouched instead of poisoning it with NaN.
            with np.errstate(divide='ignore', invalid='ignore'):
                batch_mean = np.where(
                    batch_size > 0, np.bincount(idx, w * I, minlength=bins) / batch_size, 0.
                )
                size = size + batch_size
                mean = mean + np.where(size > 0, batch_size / size, 0.) * (batch_mean - mean)

        return mean, size, labels, edges

    @classmethod
    def with_empirical_sigma(cls, rac, dataset, bins=20, maxiter=100, isigi_cutoff=0.0, standardize=True, interpolate=True, **kwargs):
        mean, size, labels, edges = cls._empirical_binned_mean(
            rac, dataset, bins=bins, maxiter=maxiter, isigi_cutoff=isigi_cutoff
        )

        if standardize:
            k = np.sum(
                mean * size / np.sum(size)
            )
            mean = mean / k

        if interpolate:
            from scipy.interpolate import interp1d
            eps = 1e-3
            x = np.concatenate((
                [edges[0]+eps], 
                0.5 * (edges[1:] + edges[:-1]),
                [edges[-1]-eps],
            ))
            y = np.concatenate(([mean[0]], mean, [mean[-1]]))
            dHKL = rac.dHKL.cpu().numpy()
            sigma = interp1d(x**-2., y)(dHKL**-2)
        else:
            sigma = mean[labels]

        return cls(rac, sigma.astype('float32'), **kwargs)


class AutoWilsonPriorBase(WilsonPriorBase):
    """
    A Wilson prior whose Σ falls off with resolution, Σ = k exp(-b / d**2), with
    b and optionally k learned. b is reported as the Wilson B of the intensities,
    2b in Å² (the metric WilsonB).
    """
    def __init__(self, *args, learn_k=True, **kwargs):
        super().__init__(*args, **kwargs)
        # b in units of 10 Å²: Adam moves a parameter about lr per step, and a
        # typical b (5-15 Å²) is out of reach of 1e-3 per step in 10k steps.
        self._b = torch.nn.Parameter(torch.tensor(0.))
        if learn_k:
            self._k = torch.nn.Parameter(torch.tensor(0.))
        else:
            self.register_buffer('_k', torch.tensor(0.))

    @property
    def b(self):
        return 10. * self._b

    @property
    def k(self):
        return torch.exp(self._k)

    @property
    def sigma(self):
        tracking.add_metric(self.k, 'WilsonK')
        tracking.add_metric(2. * self.b, 'WilsonB')
        sigma = torch.exp(-self.b * torch.reciprocal(torch.square(self.rac.dHKL)) + self._k)
        return sigma
