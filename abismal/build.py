"""
Build an abismal model and its optimizer from command-line arguments.

This is the one place that turns arguments into a model. Training calls it, and
so does `abismal.saving.load_model`, which rebuilds a checkpoint from the
arguments and data facts saved in it before loading its weights. Model classes
therefore need no serialization methods of their own: anything learned or
computed from the data lives in a parameter or buffer, and the state_dict
carries it.
"""
import logging

logger = logging.getLogger(__name__)

FACT_KEYS = ("cell", "spacegroup", "dmin", "num_asus", "metadata_length")


def facts_from(dm, train):
    """
    The properties of the data the model's structure depends on, as plain types:
    the unit cell, space group, resolution limit, number of ASUs and metadata
    length.
    """
    return {
        "cell": list(dm.cell.parameters),
        "spacegroup": dm.spacegroup.xhm(),
        "dmin": float(dm.dmin),
        "num_asus": int(dm.num_asus),
        "metadata_length": int(train.metadata_length),
    }


def build_model(args, facts, data=None, precision="float32"):
    """
    Build the `VariationalMergingModel` that `args` describe for data with these
    `facts` (see `facts_from`).

    `data` is the training stream, used only to initialize an empirical Wilson
    prior's Σ. Without it that Σ is a placeholder, which loading a state_dict
    replaces. The arguments and facts are kept on the model as `model.spec`, so
    that `abismal.saving.save_model` can write them next to the weights.
    """
    import gemmi
    import torch
    from abismal.symmetry import ReciprocalASU, ReciprocalASUGraph, merge_dmin
    from abismal.merging import VariationalMergingModel
    from abismal.scaling import ImageScaler
    from abismal.likelihood import StudentTLikelihood, NormalLikelihood

    cell = gemmi.UnitCell(*facts["cell"])
    spacegroup = gemmi.SpaceGroup(facts["spacegroup"])
    dmin = facts["dmin"]
    num_asus = facts["num_asus"]

    reindexing_ops = ["x,y,z"]
    if not args.disable_index_disambiguation:
        ops = gemmi.find_twin_laws(cell, spacegroup, 3.0, False)
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops]
        logger.info(f"Adding disambiguation operators: {reindexing_ops}")

    # A pseudo-symmetric reindexing operator moves reflections in resolution, so
    # the outermost ones map past dmin and out of the ASU. Build the ASU to
    # wherever the operators can reach and truncate back to dmin when saving.
    # This is a no-op for the identity and for true lattice symmetries.
    merge_dmin_ = merge_dmin(cell, reindexing_ops, dmin)
    if merge_dmin_ < dmin:
        logger.info(
            f"Reindexing operators reach {merge_dmin_:.4f} A from a dmin of "
            f"{dmin} A; merging at {merge_dmin_:.4f} A and truncating to "
            f"{dmin} A on output."
        )

    rasu = []
    anomalous = False if args.separate_friedel_mates else args.anomalous
    logger.info(f"Data are anomalous (True/False): {anomalous}")
    for i in range(num_asus):
        logger.info(f"Adding asu ID: {i}")
        rasu.append(
            ReciprocalASU(
                cell,
                spacegroup,
                merge_dmin_,
                anomalous=anomalous,
            )
        )

    logger.info("Combining reciprocal ASUs as collection")
    rac = ReciprocalASUGraph(
        *rasu,
        parents=args.parents,
        reindexing_ops=args.reindexing_ops,
    )

    posterior_kwargs = {}

    standardize_inputs = True
    if args.prior_distribution in ["wilson", "empirical_wilson"]:
        if args.parents is not None:
            from abismal.prior.structure_factor.wilson import MultiWilsonPrior

            prior = MultiWilsonPrior(
                rac,
                args.prior_correlation,
            )
            posterior_kwargs['independent'] = False
        elif args.posterior_type == "intensity":
            from abismal.prior.intensity.wilson import WilsonPrior
            if args.prior_distribution == 'empirical_wilson':
                prior = WilsonPrior.with_empirical_sigma(rac, data, standardize=standardize_inputs) if data is not None else WilsonPrior(rac)
            else:
                prior = WilsonPrior(rac)
        else:
            from abismal.prior.structure_factor.wilson import WilsonPrior
            if args.prior_distribution == 'empirical_wilson':
                prior = WilsonPrior.with_empirical_sigma(rac, data, standardize=standardize_inputs) if data is not None else WilsonPrior(rac)
            else:
                prior = WilsonPrior(rac)
        loc_init = prior.flat_distribution().mean
        #scale_init = args.init_scale * loc_init
        scale_init = prior.flat_distribution().stddev
        scale_init = args.init_scale * scale_init
    elif args.prior_distribution == 'auto_wilson':
        from abismal.prior.structure_factor.wilson import AutoWilsonPrior
        prior = AutoWilsonPrior(rac)
        loc_init = prior.flat_distribution().mean
        scale_init = args.init_scale * loc_init
    elif args.prior_distribution == "normal":
        from abismal.prior.normal import NormalPrior

        prior = NormalPrior(rac)
        loc_init = torch.ones_like(prior.flat_distribution().mean)
        scale_init = args.init_scale * loc_init
        if args.posterior_rank > 1:
            from abismal.prior.normal import MultivariateNormalPrior

            prior = MultivariateNormalPrior(rac)
    elif args.prior_distribution == "halfnormal":
        from abismal.prior.normal import HalfNormalPrior
        prior = HalfNormalPrior(rac)
        loc_init = torch.ones_like(prior.flat_distribution().mean)
        scale_init = args.init_scale * loc_init

    posterior_kwargs.update({
        "rac": rac,
        "loc_init": loc_init,
        "scale_init": scale_init,
        "epsilon": args.epsilon,
    })

    if args.posterior_type == "intensity":
        if args.posterior_distribution == "foldednormal":
            from abismal.surrogate_posterior.intensity import (
                FoldedNormalPosterior as Posterior,
            )
        elif args.posterior_distribution == "rice":
            raise ValueError("Rice distributed intensity posteriors are not supported.")
        elif args.posterior_distribution == "gamma":
            from abismal.surrogate_posterior.intensity.gamma import (
                GammaPosterior as Posterior,
            )
        elif args.posterior_distribution == "lognormal":
            from abismal.surrogate_posterior.intensity.lognormal import (
                LogNormalPosterior as Posterior,
            )
        elif args.posterior_distribution == "nakagami":
            raise ValueError("Nakagami distributed intensity posteriors are not supported.")
        elif args.posterior_distribution == "normal":
            if args.posterior_rank == 1:
                from abismal.surrogate_posterior.intensity.normal import (
                    NormalPosterior as Posterior,
                )
            else:
                from abismal.surrogate_posterior.intensity.normal import (
                    MultivariateNormalPosterior as Posterior,
                )

                posterior_kwargs["rank"] = args.posterior_rank
    elif args.posterior_type == "structure_factor":
        if args.posterior_distribution == "foldednormal":
            from abismal.surrogate_posterior.structure_factor.folded_normal import (
                FoldedNormalPosterior as Posterior,
            )
        elif args.posterior_distribution == "rice":
            from abismal.surrogate_posterior.structure_factor.rice import (
                RicePosterior as Posterior,
            )
        elif args.posterior_distribution == "nakagami":
            from abismal.surrogate_posterior.structure_factor.nakagami import (
                NakagamiPosterior as Posterior,
            )
        elif args.posterior_distribution == "lognormal":
            from abismal.surrogate_posterior.structure_factor.lognormal import (
                LogNormalPosterior as Posterior,
            )
        elif args.posterior_distribution == "normal":
            if args.posterior_rank == 1:
                from abismal.surrogate_posterior.structure_factor.normal import (
                    NormalPosterior as Posterior,
                )
            else:
                from abismal.surrogate_posterior.structure_factor.normal import (
                    MultivariateNormalPosterior as Posterior,
                )

                posterior_kwargs["rank"] = args.posterior_rank

    surrogate_posterior = Posterior(**posterior_kwargs)

    scale_model = ImageScaler(
        metadata_length=facts["metadata_length"],
        mlp_width=args.d_model,
        mlp_depth=args.layers,
        hidden_units=max(1, round(args.d_model * args.hidden_multiplier)),
        activation=args.activation,
        kl_weight=args.scale_kl_weight,
        epsilon=args.epsilon,
        ff_epsilon=args.ff_epsilon,
        prior_name=args.scale_prior_distribution,
        posterior_name=args.scale_posterior_distribution,
        bijector_name=args.scale_posterior_bijector,
        pre_activation=args.pre_activation,
        gated=args.gated,
        dropout=args.dropout,
        batch_normalize=args.batch_normalize,
        metadata_noise_factor=args.metadata_noise_factor,
        precision=precision,
        share_weights=True,
        posterior_features=2 if args.posterior_features else 0,
    )

    if args.studentt_dof is not None:
        likelihood = StudentTLikelihood(args.studentt_dof)
    else:
        likelihood = NormalLikelihood()

    model = VariationalMergingModel(
        scale_model,
        surrogate_posterior,
        prior=prior,
        likelihood=likelihood,
        mc_samples=args.mc_samples,
        kl_weight=args.kl_weight,
        reindexing_ops=reindexing_ops,
        standardize=standardize_inputs,
    )

    model.spec = {"args": dict(vars(args)), "facts": dict(facts)}
    return model


def build_optimizer(args, model):
    """The optimizer `args` describe, over `model`'s trainable parameters."""
    from abismal.training import parameter_groups, piecewise_constant

    if args.learning_rate_final is not None:
        # The TF version built this schedule and then handed the optimizer the
        # constant rate anyway, so --learning-rate-final did nothing.
        learning_rate = piecewise_constant(
            args.burnin, args.learning_rate, args.learning_rate_final
        )
    else:
        learning_rate = args.learning_rate

    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "beta_1": args.beta_1,
        "beta_2": args.beta_2,
        "epsilon": args.adam_epsilon,
        "clipnorm": args.clipnorm,
        "clipvalue": args.clip,
        "global_clipnorm": args.global_clipnorm,
    }
    from abismal.optimizers.optimizer_dict import optimizer_dict

    # Passed only when set, so every optimizer keeps its own default -- notably
    # AdamW's 0.004, which an unconditional None would silently turn back into
    # plain Adam.
    if args.weight_decay is not None:
        optimizer_kwargs['weight_decay'] = args.weight_decay

    Optimizer = optimizer_dict[args.optimizer]

    return Optimizer(parameter_groups(model), **optimizer_kwargs)
