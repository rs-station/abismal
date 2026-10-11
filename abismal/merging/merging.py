import torch
import torch.nn.functional as F
from abismal import serialization
from abismal import tracking
from abismal.symmetry import Op
from abismal.layers.normalization import StandardizeIntensities,StandardizeMetadata
from abismal.scaling.scaling import row_weights


def _per_image_sum(values, image_id, num_images):
    """Sum per-reflection `values`, shape (reflections, ...), within each image."""
    out = values.new_zeros((num_images,) + values.shape[1:])
    return out.index_add_(0, image_id, values)


@serialization.register
class VariationalMergingModel(torch.nn.Module, serialization.Serializable):
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
            background=None,
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
        # Experiment (background model). None, or
        #   'tau'      : sigma^2 -> sigma^2 + tau^2
        #   'analytic' : ipred += tau * u, u from the scale network, penalty u^2/2,
        #                and sigma^2 -> sigma^2 + tau^2
        #   'latent'   : ipred += B ~ q = N(tau m, tau s), (m, s) from the scale
        #                network, KL(q || N(0, tau^2)); sigma unchanged
        # tau is the last metadata column, in counts.
        if background not in (None, 'tau', 'analytic', 'latent'):
            raise ValueError(f"unknown background mode {background!r}")
        self.background = background
        if standardize:
            self.standardize_intensity = StandardizeIntensities()
            self.standardize_metadata = StandardizeMetadata(scale_model.metadata_length)
        else:
            self.standardize_intensity = None
            self.standardize_metadata = None

    def get_config(self):
        config = {
            'scale_model' : serialization.serialize(self.scale_model),
            'surrogate_posterior' : serialization.serialize(self.surrogate_posterior),
            'prior' : serialization.serialize(self.prior),
            'likelihood' : serialization.serialize(self.likelihood),
            'mc_samples' : self.mc_samples,
            'kl_weight' : self.kl_weight,
            'epsilon' : self.epsilon,
            'reindexing_ops' : [op.triplet() for op in self.reindexing_ops],
            'standardize' : self.standardize,
            'background' : self.background,
        }
        return config

    @classmethod
    def from_config(cls, config):
        for k in ['scale_model', 'surrogate_posterior', 'likelihood', 'prior']:
            config[k] = serialization.deserialize(config[k])
        return cls(**config)

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
        if getattr(inputs, "context_weight", None) is not None or getattr(inputs, "loss_weight", None) is not None:
            raise NotImplementedError("Row weights are only implemented for independent posteriors")
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

        tau = None
        if self.background is not None:
            tau = inputs.metadata[:, -1:].float()

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
        bg_params = None
        if isinstance(scale, tuple):
            scale, bg_params = scale

        context, loss_weight = row_weights(inputs)

        b, kl_b, sig_ll = 0., None, sigiobs
        if self.background is not None:
            if self.standardize_intensity is not None:
                tau = self.standardize_intensity.standardize(tau)
            if self.background in ('tau', 'analytic'):
                sig_ll = torch.sqrt(sigiobs * sigiobs + tau * tau)
            if self.background == 'analytic':
                u = bg_params[:, :1]
                b = tau * u
                kl_b = 0.5 * u[:, 0] ** 2
            elif self.background == 'latent':
                m = bg_params[:, :1]
                # softplus(0.5413) = 1: q starts equal to the prior
                s = F.softplus(bg_params[:, 1:2] + 0.5413) + 1e-4
                b = tau * (m + s * torch.randn(m.shape[0], mc_samples, device=m.device, dtype=m.dtype))
                kl_b = (0.5 * (m * m + s * s - 1.) - torch.log(s))[:, 0]
                tracking.add_metric(s.mean(), 'B_s')
            if self.background != 'tau':
                bmean = b.mean(-1) if b.dim() > 1 and b.shape[-1] > 1 else b[:, 0]
                tracking.add_metric((bmean / sigiobs[:, 0]).mean(), 'B/σ')
                tracking.add_metric((bmean / sigiobs[:, 0]).abs().mean(), '|B|/σ')

        score = None
        ipred = None
        hkl = None
        kl_div = None
        ll = None
        valid = None

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
            _ipred = _ipred * scale + b

            _ll = self.likelihood(_ipred, iobs, sig_ll)

            _valid = self.surrogate_posterior.rac.valid(asu_id, _hkl)
            _kl_div = _kl_div * _valid.to(_kl_div.dtype)
            # Each image's operator is chosen on the reflections that inform it
            _score = self._image_log_likelihood(
                _ll, _valid & (context > 0)[:, None], image_id, num_images
            )

            idx = None if score is None else _score > score
            ipred = self._select(ipred, idx, image_id, _ipred)
            hkl = self._select(hkl, idx, image_id, _hkl)
            kl_div = self._select(kl_div, idx, image_id, _kl_div)
            ll = self._select(ll, idx, image_id, _ll)
            valid = self._select(valid, idx, image_id, _valid)
            score = _score if score is None else torch.where(idx, _score, score)

        if training:
            self.surrogate_posterior.register_seen(asu_id, hkl)

        self.likelihood.register_metrics(ipred, iobs, sigiobs, weight=loss_weight)

        # The objective: each image's mean log likelihood over its counted
        # reflections and the Monte Carlo samples, averaged over images. An
        # image with nothing to count (none valid, or all of its reflections
        # given loss weight 0) is left out of the average.
        w = valid[:, 0].to(ll.dtype) * loss_weight
        counted = _per_image_sum(w, image_id, num_images)
        total = _per_image_sum(ll.sum(-1) * w, image_id, num_images)
        image_ll = total / (counted * ll.shape[-1]).clamp_min(1.)
        has = (counted > 0).to(ll.dtype)
        ll = (image_ll * has).sum() / has.sum().clamp_min(1.)

        tracking.add_metric(-ll, name='NLL')
        tracking.add_loss(-ll)

        kl_div = (kl_div[:, 0] * loss_weight).sum() / loss_weight.sum().clamp_min(1.)
        tracking.add_metric(kl_div, name='KL')
        tracking.add_loss(self.kl_weight * kl_div)

        if kl_b is not None:
            kl_b = (kl_b * w).sum() / w.sum().clamp_min(1.)
            tracking.add_metric(kl_b, name='KL_B')
            tracking.add_loss(kl_b)

        ipred_avg = ipred.mean(-1)
        return ipred_avg
