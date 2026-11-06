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
            kl_reflection_weights='none',
        ):
        super().__init__()
        self.epsilon = epsilon
        self.likelihood = likelihood
        self.prior = prior
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.mc_samples = mc_samples
        self.kl_weight = kl_weight
        self.kl_reflection_weights = kl_reflection_weights
        if kl_reflection_weights != 'none':
            # Observations of each reflection, each image counted once (Batch.first_pass).
            self.register_buffer('kl_counts', torch.zeros(surrogate_posterior.rac.asu_size, dtype=torch.float64))
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
            # Before "q", whose parameters update lazily: the trend's are shared by every reflection.
            ("trend", self.surrogate_posterior.trend),
            ("q", self.surrogate_posterior),
            ("ll", self.likelihood),
            ("p", self.prior),
        ]:
            params = [] if module is None else [
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

    def _posterior_features(self, asu_id, hkl, dtype):
        """
        The current posterior mean and standard deviation of each reflection under
        the identity operator, for a scale model that takes them, else None.

        They carry no gradient: the scale model reads the estimates but does not
        move them. Reflections outside the ASU get zeros rather than the stand-in
        posterior `gather` would give them.
        """
        if not getattr(self.scale_model, 'posterior_features', 0):
            return None
        with torch.no_grad():
            posterior = self.surrogate_posterior
            if posterior.independent:
                q = posterior.distribution(asu_id, hkl)
                features = torch.stack([q.mean, q.stddev], -1).reshape(-1, 2)
            else:
                q = posterior.flat_distribution()
                features = posterior.rac.gather(torch.stack([q.mean, q.stddev], -1), asu_id, hkl).reshape(-1, 2)
            valid = posterior.rac.valid(asu_id, hkl).reshape(-1, 1)
            return features.to(dtype) * valid.to(dtype)

    def _image_log_likelihood(self, ll, valid, image_id, num_images):
        """
        Average log likelihood of each image over the reflections a reindexing
        operator can actually index and over the Monte Carlo samples, used to
        choose each image's operator (the loss is per observation, see
        _observation_mean).

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

    @torch.no_grad()
    def _count_observations(self, asu_id, hkl, first_pass):
        """Add the batch's images that are new to the stream to the reflections' observation counts."""
        rac = self.surrogate_posterior.rac
        ids = rac._miller_ids(asu_id, hkl).reshape(-1)
        ids = torch.where(ids >= 0, ids, rac.asu_size)  # a sentinel slot past the end, as in register_seen
        counts = torch.zeros(rac.asu_size + 1, dtype=self.kl_counts.dtype, device=ids.device)
        counts.index_add_(0, ids, first_pass.reshape(-1).to(counts.dtype))
        self.kl_counts += counts[:-1]

    @torch.no_grad()
    def _kl_reflection_weight(self, asu_id, hkl):
        """
        Each observation's weight on its reflection's KL. 'exact': 1/n_h, so that a reflection's prior counts once
        against its n_h likelihood terms, the ELBO itself. 'standardized': 1/n_h over its mean over the reflections seen
        so far, so that --kl-weight sets the overall strength. 1 for everything before any reflection has been counted.
        """
        seen = self.kl_counts > 0
        inverse = 1. / self.kl_counts.clamp_min(1.)
        if self.kl_reflection_weights == 'standardized':
            inverse = inverse / ((inverse * seen).sum() / seen.sum().clamp_min(1)).clamp_min(1e-30)
        weight = torch.where(seen.any(), inverse, torch.ones_like(inverse))
        return self.surrogate_posterior.rac.gather(weight, asu_id, hkl)

    def _observation_mean(self, x, valid):
        """
        The mean of a per-observation term over the valid observations of the chosen operators and its trailing
        (Monte Carlo) axis: the objective's log likelihood and, for independent posteriors, its KL, so every
        observation counts once whatever image it is on and observations nothing indexes count not at all. (The
        per-image means of _image_log_likelihood only choose each image's operator.)
        """
        w = valid.to(x.dtype)
        return (x * w).sum() / (w.sum() * x.shape[-1]).clamp_min(1.)

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
            features=self._posterior_features(asu_id, hkl_in, iobs.dtype),
        )

        ll = None
        image_ll = None
        valid = None
        ipred = None
        hkl = None
        q = self.surrogate_posterior.flat_distribution()
        z = q.rsample((mc_samples,))

        p = self.prior.flat_distribution()
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
            if self.surrogate_posterior.trend is not None:
                _ipred = _ipred * self.surrogate_posterior.trend(asu_id, _hkl)
            _ipred = _ipred * scale

            _ll = self.likelihood(_ipred, iobs, sigiobs)

            _valid = self.surrogate_posterior.rac.valid(asu_id, _hkl)
            _image_ll = self._image_log_likelihood(_ll, _valid, image_id, num_images)

            idx = None if image_ll is None else _image_ll > image_ll
            ll = self._select(ll, idx, image_id, _ll)
            ipred = self._select(ipred, idx, image_id, _ipred)
            valid = self._select(valid, idx, image_id, _valid)
            hkl = self._select(hkl, idx, image_id, _hkl)
            image_ll = _image_ll if image_ll is None else torch.where(idx, _image_ll, image_ll)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)

        if training and getattr(self.likelihood, 'fit', None) == 'online':
            self.likelihood.online_update(ipred, iobs, sigiobs, valid)
        self.likelihood.register_metrics(ipred, iobs, sigiobs)

        ll = self._observation_mean(ll, valid)
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
            features=self._posterior_features(asu_id, hkl_in, iobs.dtype),
        )

        ll = None
        image_ll = None
        valid = None
        ipred = None
        hkl = None
        kl_div = None

        for op in self.reindexing_ops:
            _hkl = op(hkl_in)
            q = self.surrogate_posterior.distribution(asu_id, _hkl)
            p = self.prior.distribution(asu_id, _hkl)
            z = q.rsample((mc_samples,))
            _kl_div = self.surrogate_posterior.compute_kl_terms(q, p, samples=z)

            _kl_div = _kl_div[..., None]
            _ipred = z.transpose(0, 1)

            if self.surrogate_posterior.parameterization == 'structure_factor':
                _ipred = torch.square(_ipred)
            if self.surrogate_posterior.trend is not None:
                _ipred = _ipred * self.surrogate_posterior.trend(asu_id, _hkl)
            _ipred = _ipred * scale

            _ll = self.likelihood(_ipred, iobs, sigiobs)

            _valid = self.surrogate_posterior.rac.valid(asu_id, _hkl)
            _kl_div = _kl_div * _valid.to(_kl_div.dtype)
            if self.kl_reflection_weights != 'none':
                _kl_div = _kl_div * self._kl_reflection_weight(asu_id, _hkl).reshape(_kl_div.shape).to(_kl_div.dtype)
            _image_ll = self._image_log_likelihood(_ll, _valid, image_id, num_images)

            idx = None if image_ll is None else _image_ll > image_ll
            ll = self._select(ll, idx, image_id, _ll)
            ipred = self._select(ipred, idx, image_id, _ipred)
            valid = self._select(valid, idx, image_id, _valid)
            hkl = self._select(hkl, idx, image_id, _hkl)
            kl_div = self._select(kl_div, idx, image_id, _kl_div)
            image_ll = _image_ll if image_ll is None else torch.where(idx, _image_ll, image_ll)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)
            if self.kl_reflection_weights != 'none':
                self._count_observations(asu_id, hkl, inputs.first_pass)

        if training and getattr(self.likelihood, 'fit', None) == 'online':
            self.likelihood.online_update(ipred, iobs, sigiobs, valid)
        self.likelihood.register_metrics(ipred, iobs, sigiobs)

        ll = self._observation_mean(ll, valid)

        tracking.add_metric(-ll, name='NLL')
        tracking.add_loss(-ll)

        kl_div = self._observation_mean(kl_div, valid)
        tracking.add_metric(kl_div, name='KL')
        tracking.add_loss(self.kl_weight * kl_div)

        ipred_avg = ipred.mean(-1)
        return ipred_avg
