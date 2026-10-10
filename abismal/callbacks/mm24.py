"""
Calibrate the error model from pairwise differences of equivalent observations,
after Mittan-Moreau et al. (2025) (`error_modifier_mm24` in cctbx.xfel.merge).

Before training and after every epoch, one inference pass puts each training
observation on a common scale with the scale model's posterior mean, I/Sigma
and sigma/Sigma. For every pair (i, j) of observations of the same reflection,
the true intensity cancels from the difference,

    d_ij = I_i/Sigma_i - I_j/Sigma_j  ~  t_nu(0, v_i + v_j),
    v = SdFac^2 (sigma^2/Sigma^2 + SdAdd^2 Ibar^2),

where Ibar is the reflection's mean scaled intensity (the "biased mean"). SdFac
and SdAdd are fit by maximum likelihood and frozen into the model's
`CalibratedLikelihood` for the next epoch. Both are dimensionless, so the same
values hold on the raw scale, with Sigma Ibar ~ mu, the model's expected
intensity. The calibration never sees the structure factors.
"""
import os
import numpy as np
import torch
from abismal.callbacks.base import Callback
from abismal.io.pipeline import BatchStream


class Mm24Calibrator(Callback):
    def __init__(self, train, out_dir=None, degrees_of_freedom=10., max_pairs=100, seed=0, at_start=True):
        self.train = train
        self.out_dir = out_dir
        self.degrees_of_freedom = degrees_of_freedom
        self.max_pairs = max_pairs
        self.rng = np.random.default_rng(seed)
        self.at_start = at_start
        self.sdfac, self.sdadd = 1., 0.05

    @torch.no_grad()
    def scaled_observations(self, update_standardization=False):
        model, trainer = self.model, self.trainer
        stream = BatchStream(self.train.cache, self.train.split, batch_size=self.train.batch_size, repeat=False)
        rac = model.surrogate_posterior.rac
        ids, i_s, s_s = [], [], []
        for batch in stream.iterate(transfer=trainer.transfer, prefetch=2):
            inputs = model.standardize_inputs(batch, training=update_standardization)
            sigma = model.scale_model(inputs, mc_samples=model.mc_samples, training=False).mean(-1)
            idx = rac._miller_ids(inputs.asu_id, inputs.hkl)
            keep = idx >= 0
            ids.append(idx[keep].cpu())
            i_s.append((inputs.iobs[:, 0] / sigma)[keep].double().cpu())
            s_s.append((inputs.sigiobs[:, 0] / sigma)[keep].double().cpu())
        return torch.cat(ids).numpy(), torch.cat(i_s).numpy(), torch.cat(s_s).numpy()

    def pairs(self, ids):
        """Up to `max_pairs` random pairs (i != j) of observations of each reflection."""
        order = np.argsort(ids, kind='stable')
        _, start, n = np.unique(ids[order], return_index=True, return_counts=True)
        multi = n > 1
        start, n = start[multi], n[multi]
        npairs = np.minimum(n * (n - 1) // 2, self.max_pairs)
        g_start = np.repeat(start, npairs)
        g_n = np.repeat(n, npairs)
        a = (self.rng.random(g_n.size) * g_n).astype(np.int64)
        b = (a + 1 + (self.rng.random(g_n.size) * (g_n - 1)).astype(np.int64)) % g_n
        return order[g_start + a], order[g_start + b], order, start, n

    def fit(self, ids, i_s, s_s):
        i, j, order, start, n = self.pairs(ids)
        # Biased mean of each observation's reflection
        sums = np.add.reduceat(i_s[order], np.unique(ids[order], return_index=True)[1])
        counts = np.diff(np.append(np.unique(ids[order], return_index=True)[1], ids.size))
        ibar_sorted = np.repeat(sums / counts, counts)
        ibar = np.empty_like(i_s)
        ibar[order] = ibar_sorted

        d = torch.as_tensor(i_s[i] - i_s[j])
        s2 = torch.as_tensor(s_s[i] ** 2 + s_s[j] ** 2)
        ib2 = torch.as_tensor(ibar[i] ** 2)
        nu = self.degrees_of_freedom
        x = torch.tensor([np.log(self.sdfac), np.log(max(self.sdadd, 1e-4))], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([x], max_iter=200, line_search_fn='strong_wolfe', tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            sdfac, sdadd = x.exp()
            var = sdfac ** 2 * (s2 + 2. * sdadd ** 2 * ib2)
            nll = 0.5 * torch.log(var) + 0.5 * (nu + 1.) * torch.log1p(d ** 2 / (nu * var))
            loss = nll.mean()
            loss.backward()
            return loss
        opt.step(closure)
        sdfac, sdadd = x.detach().exp().tolist()
        return sdfac, sdadd, d.numel()

    def calibrate(self, epoch, update_standardization=False):
        ids, i_s, s_s = self.scaled_observations(update_standardization)
        self.sdfac, self.sdadd, npairs = self.fit(ids, i_s, s_s)
        self.model.likelihood.set_parameters(sdfac=self.sdfac, sdb=0., sdadd=self.sdadd)
        print(f"MM24 calibration after epoch {epoch}: SdFac {self.sdfac:.4f} SdAdd {self.sdadd:.4f} "
              f"({npairs} pairs from {ids.size} observations)", flush=True)
        if self.out_dir is not None:
            path = os.path.join(self.out_dir, "mm24.csv")
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("epoch,sdfac,sdadd,pairs,observations\n")
                f.write(f"{epoch},{self.sdfac},{self.sdadd},{npairs},{ids.size}\n")

    def on_train_begin(self, logs=None):
        if self.at_start:
            # The standardization layers have seen nothing yet; let this pass
            # feed them, as the first training steps would.
            self.calibrate(0, update_standardization=True)

    def on_epoch_end(self, epoch, logs=None):
        self.calibrate(epoch + 1)
