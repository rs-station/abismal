import torch
from abismal import tracking
from abismal.symmetry import Op
from abismal.layers.normalization import StandardizeIntensities,StandardizeMetadata


def _per_image_sum(values, image_id, num_images):
    """Sum per-reflection `values`, shape (reflections, ...), within each image."""
    out = values.new_zeros((num_images,) + values.shape[1:])
    return out.index_add_(0, image_id, values)
class VariationalMergingModel(torch.nn.Module):
    def __init__(
            self,
            scale_model,
            surrogate_posterior,
            prior,
            likelihood,
            mc_samples=1,
            kl_weight=1.,
            epsilon=1e-6,
            reindexing_ops=None,
            standardize=False,
        ):
        super().__init__()
        self.epsilon = epsilon
        self.likelihood = likelihood
        self.prior = prior
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.mc_samples = mc_samples
        self.kl_weight = kl_weight
        if reindexing_ops is None:
            reindexing_ops = ["x,y,z"]
        self.reindexing_ops = [Op(op) for op in reindexing_ops]
        self.standardize = standardize
        if standardize:
            self.standardize_intensity = StandardizeIntensities()
            self.standardize_metadata = StandardizeMetadata(scale_model.metadata_length)
        else:
            self.standardize_intensity = None
            self.standardize_metadata = None

    def parameter_groups(self):
        """
        The trainable parameters by component, for per-component gradient
        norms and for the optimizer, which updates the surrogate posterior
        lazily. Shared modules are reported once, under the first component
        that owns them.
        """
        groups = {}
        seen = set()
        for name, module in [
            ("scale", self.scale_model),
            ("q", self.surrogate_posterior),
            ("ll", self.likelihood),
            ("p", self.prior),
        ]:
            params = [
                p for p in module.parameters()
                if p.requires_grad and id(p) not in seen
            ]
            seen.update(id(p) for p in params)
            groups[name] = params
        return groups

    def standardize_inputs(self, inputs, training=None):
        out = inputs
        if self.standardize_intensity is not None:
            out = self.standardize_intensity(out, training=training)
        if self.standardize_metadata is not None:
            out = self.standardize_metadata(out, training=training)
        return out

    def forward(self, inputs, mc_samples=None, training=None):
        """
        Parameters
        ----------
        inputs : abismal.io.Batch
            A batch of images as flat, per-reflection tensors.
        mc_samples : int (optional)
        training : bool (optional)

        Returns
        -------
        ipred : torch.Tensor
            The predicted intensity of each reflection, averaged over the Monte
            Carlo samples, shape (reflections,).

        The negative log likelihood and KL divergence are reported through
        `abismal.tracking`.
        """
        if self.surrogate_posterior.independent:
            return self._call_independent(inputs, mc_samples=mc_samples, training=training)
        return self._call_dependent(inputs, mc_samples=mc_samples, training=training)

    def _image_log_likelihood(self, ll, valid, image_id, num_images):
        """
        Average log likelihood of each image over the reflections a reindexing
        operator can actually index and over the Monte Carlo samples.

        An operator can push a reflection past dmin and out of the ASU, where
        the gather silently returns reflection 0 instead. Score the operators
        only on what they can actually index, so a miss cannot earn one of them
        a spurious likelihood; an operator that indexes nothing in an image
        loses outright.
        """
        valid = valid.to(ll.dtype)
        ll = ll * valid
        nvalid = _per_image_sum(valid[:, 0], image_id, num_images)
        mc = ll.shape[-1]
        total = _per_image_sum(ll.sum(-1), image_id, num_images)
        image_ll = torch.where(nvalid > 0., total / (nvalid * mc).clamp_min(1.), -1e30)
        return image_ll

    def _select(self, best, idx, image_id, candidate):
        """Take `candidate` where this operator won its image, `best` elsewhere."""
        if best is None:
            return candidate
        pick = idx[image_id]
        pick = pick.reshape(pick.shape + (1,) * (candidate.dim() - 1))
        return torch.where(pick, candidate, best)

    def _call_dependent(self, inputs, mc_samples=None, training=None):
        """
        Used for posteriors in which different miller indices are statistically dependent such as the MultivariateNormalPosterior
        """
        if mc_samples is None:
            mc_samples = self.mc_samples

        inputs = self.standardize_inputs(inputs, training=training)

        asu_id = inputs.asu_id
        hkl_in = inputs.hkl
        iobs = inputs.iobs
        sigiobs = inputs.sigiobs
        image_id = inputs.image_id
        num_images = inputs.num_images

        scale = self.scale_model(
            inputs,
            mc_samples=mc_samples,
            training=training,
        )

        ll = None
        ipred = None
        hkl = None
        q = self.surrogate_posterior.flat_distribution()
        z = q.rsample((mc_samples,))

        p = self.prior(inputs, flat=True)
        kl_div = self.surrogate_posterior.compute_kl_terms(q, p, samples=z)

        for op in self.reindexing_ops:
            _hkl = op(hkl_in)
            _ipred = self.surrogate_posterior.rac.gather(
                z.transpose(0, 1),
                asu_id,
                _hkl,
            )

            if self.surrogate_posterior.parameterization == 'structure_factor':
                _ipred = torch.square(_ipred)
            _ipred = _ipred * scale

            _ll = self.likelihood(_ipred, iobs, sigiobs)

            _valid = self.surrogate_posterior.rac.valid(asu_id, _hkl)
            _ll = self._image_log_likelihood(_ll, _valid, image_id, num_images)

            idx = None if ll is None else _ll > ll
            ipred = self._select(ipred, idx, image_id, _ipred)
            hkl = self._select(hkl, idx, image_id, _hkl)
            ll = _ll if ll is None else torch.where(idx, _ll, ll)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)

        self.likelihood.register_metrics(ipred, iobs, sigiobs)

        ll = ll.mean()
        kl_div = kl_div.mean()

        tracking.add_metric(-ll, name='NLL')
        tracking.add_loss(-ll)

        tracking.add_metric(kl_div, name='KL')
        tracking.add_loss(self.kl_weight * kl_div)

        ipred_avg = ipred.mean(-1)
        return ipred_avg

    def _call_independent(self, inputs, mc_samples=None, training=None):
        """
        The default method can be used for most posteriors
        """
        if mc_samples is None:
            mc_samples = self.mc_samples

        inputs = self.standardize_inputs(inputs, training=training)

        asu_id = inputs.asu_id
        hkl_in = inputs.hkl
        iobs = inputs.iobs
        sigiobs = inputs.sigiobs
        image_id = inputs.image_id
        num_images = inputs.num_images

        scale = self.scale_model(
            inputs,
            mc_samples=mc_samples,
            training=training,
        )

        ll = None
        ipred = None
        hkl = None
        kl_div = None

        for op in self.reindexing_ops:
            _hkl = op(hkl_in)
            q = self.surrogate_posterior.distribution(asu_id, _hkl)
            p = self.prior(inputs._replace(hkl=_hkl), flat=False)
            z = q.rsample((mc_samples,))
            _kl_div = self.surrogate_posterior.compute_kl_terms(q, p, samples=z)

            _kl_div = _kl_div[..., None]
            _ipred = z.transpose(0, 1)

            if self.surrogate_posterior.parameterization == 'structure_factor':
                _ipred = torch.square(_ipred)
            _ipred = _ipred * scale

            _ll = self.likelihood(_ipred, iobs, sigiobs)

            _valid = self.surrogate_posterior.rac.valid(asu_id, _hkl)
            _kl_div = _kl_div * _valid.to(_kl_div.dtype)
            _ll = self._image_log_likelihood(_ll, _valid, image_id, num_images)

            idx = None if ll is None else _ll > ll
            ipred = self._select(ipred, idx, image_id, _ipred)
            hkl = self._select(hkl, idx, image_id, _hkl)
            kl_div = self._select(kl_div, idx, image_id, _kl_div)
            ll = _ll if ll is None else torch.where(idx, _ll, ll)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)

        self.likelihood.register_metrics(ipred, iobs, sigiobs)

        ll = ll.mean()

        tracking.add_metric(-ll, name='NLL')
        tracking.add_loss(-ll)

        kl_div = kl_div.mean()
        tracking.add_metric(kl_div, name='KL')
        tracking.add_loss(self.kl_weight * kl_div)

        ipred_avg = ipred.mean(-1)
        return ipred_avg
