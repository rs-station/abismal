#!/usr/bin/env python

def main(args=None):
    from time import time
    start_time = time()
    from abismal.command_line.parser import parser

    parser = parser.parse_args(args)

    import torch
    from abismal.device import select_device, configure

    device = select_device(parser.gpu_id, parser.device)
    configure(device)
    if parser.seed is not None:
        torch.manual_seed(parser.seed)

    from abismal import __version__ as version
    from abismal.symmetry import (
        ReciprocalASU,
        ReciprocalASUCollection,
        ReciprocalASUGraph,
        merge_dmin,
    )
    from abismal.merging import VariationalMergingModel
    from abismal.callbacks import (
        HistorySaver,
        MtzSaver,
        FriedelMtzSaver,
        PhenixRunner,
        TorchRefRunner,
        AnomalousPeakFinder,
        WeightSaver,
        StandardizationFreezer,
        DifferenceMap,
    )
    from abismal.scaling import ImageScaler
    from abismal.training import Trainer, EarlyStopping, parameter_groups, piecewise_constant
    from abismal.saving import load_model
    import gemmi
    import logging
    from os.path import exists
    from os import mkdir
    from abismal.likelihood import StudentTLikelihood
    from abismal.likelihood import NormalLikelihood

    if not exists(parser.out_dir):
        mkdir(parser.out_dir)

    from abismal.io.manager import DataManager

    out_dir = str(parser.out_dir)
    log_file = out_dir + "/abismal.log"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logger.info(f"Starting abismal, version {version}")
    msg = "\n".join([f"  {k}: {v}" for k, v in vars(parser).items()])
    logger.info("Running with the following options... ")
    logger.info(msg)

    logger.info(f"Training on {device}")
    logger.info("Configuring data input")
    dm = DataManager.from_parser(parser)
    train, test = dm.get_train_test_splits()
    dm_file = out_dir + "/datamanager.yml"
    dm.to_file(dm_file)
    logger.info(f"Data manager config written to: {dm_file}")
    if test is not None:
        logger.info("There is a test set for validation")

    reindexing_ops = ["x,y,z"]
    if not parser.disable_index_disambiguation:
        ops = gemmi.find_twin_laws(dm.cell, dm.spacegroup, 3.0, False)
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops]
        logger.info(f"Adding disambiguation operators: {reindexing_ops}")

    # A pseudo-symmetric reindexing operator moves reflections in resolution, so
    # the outermost ones map past dmin and out of the ASU. Build the ASU to
    # wherever the operators can reach and truncate back to dmin when saving.
    # This is a no-op for the identity and for true lattice symmetries.
    merge_dmin_ = merge_dmin(dm.cell, reindexing_ops, dm.dmin)
    if merge_dmin_ < dm.dmin:
        logger.info(
            f"Reindexing operators reach {merge_dmin_:.4f} A from a dmin of "
            f"{dm.dmin} A; merging at {merge_dmin_:.4f} A and truncating to "
            f"{dm.dmin} A on output."
        )

    rasu = []
    anomalous = False if parser.separate_friedel_mates else parser.anomalous
    logger.info(f"Data are anomalous (True/False): {anomalous}")
    for i in range(dm.num_asus):
        logger.info(f"Adding asu ID: {i}")
        rasu.append(
            ReciprocalASU(
                dm.cell,
                dm.spacegroup,
                merge_dmin_,
                anomalous=anomalous,
            )
        )

    logger.info("Combining reciprocal ASUs as collection")
    rac = ReciprocalASUGraph(
        *rasu,
        parents=parser.parents,
        reindexing_ops=parser.reindexing_ops,
    )

    posterior_kwargs = {}

    standardize_inputs = True
    if parser.prior_distribution in ["wilson", "empirical_wilson"]:
        if parser.parents is not None:
            from abismal.prior.structure_factor.wilson import MultiWilsonPrior

            prior = MultiWilsonPrior(
                rac,
                parser.prior_correlation,
            )
            posterior_kwargs['independent'] = False
        elif parser.posterior_type == "intensity":
            from abismal.prior.intensity.wilson import WilsonPrior
            if parser.prior_distribution == 'empirical_wilson':
                prior = WilsonPrior.with_empirical_sigma(rac, train, standardize=standardize_inputs)
            else:
                prior = WilsonPrior(rac)
        else:
            from abismal.prior.structure_factor.wilson import WilsonPrior
            if parser.prior_distribution == 'empirical_wilson':
                prior = WilsonPrior.with_empirical_sigma(rac, train, standardize=standardize_inputs)
            else:
                prior = WilsonPrior(rac)
        loc_init = prior.flat_distribution().mean
        #scale_init = parser.init_scale * loc_init
        scale_init = prior.flat_distribution().stddev
        scale_init = parser.init_scale * scale_init
    elif parser.prior_distribution == 'auto_wilson':
        from abismal.prior.structure_factor.wilson import AutoWilsonPrior
        prior = AutoWilsonPrior(rac)
        loc_init = prior.flat_distribution().mean
        scale_init = parser.init_scale * loc_init
    elif parser.prior_distribution == "normal":
        from abismal.prior.normal import NormalPrior

        prior = NormalPrior(rac)
        loc_init = torch.ones_like(prior.flat_distribution().mean)
        scale_init = parser.init_scale * loc_init
        if parser.posterior_rank > 1:
            from abismal.prior.normal import MultivariateNormalPrior

            prior = MultivariateNormalPrior(rac)
    elif parser.prior_distribution == "halfnormal":
        from abismal.prior.normal import HalfNormalPrior
        prior = HalfNormalPrior(rac)
        loc_init = torch.ones_like(prior.flat_distribution().mean)
        scale_init = parser.init_scale * loc_init

    posterior_kwargs.update({
        "rac": rac,
        "loc_init": loc_init,
        "scale_init": scale_init,
        "epsilon": parser.epsilon,
    })

    if parser.posterior_type == "intensity":
        if parser.posterior_distribution == "foldednormal":
            from abismal.surrogate_posterior.intensity import (
                FoldedNormalPosterior as Posterior,
            )
        elif parser.posterior_distribution == "rice":
            raise ValueError("Rice distributed intensity posteriors are not supported.")
        elif parser.posterior_distribution == "gamma":
            from abismal.surrogate_posterior.intensity.gamma import (
                GammaPosterior as Posterior,
            )
        elif parser.posterior_distribution == "lognormal":
            from abismal.surrogate_posterior.intensity.lognormal import (
                LogNormalPosterior as Posterior,
            )
        elif parser.posterior_distribution == "nakagami":
            raise ValueError("Nakagami distributed intensity posteriors are not supported.")
        elif parser.posterior_distribution == "normal":
            if parser.posterior_rank == 1:
                from abismal.surrogate_posterior.intensity.normal import (
                    NormalPosterior as Posterior,
                )
            else:
                from abismal.surrogate_posterior.intensity.normal import (
                    MultivariateNormalPosterior as Posterior,
                )

                posterior_kwargs["rank"] = parser.posterior_rank
    elif parser.posterior_type == "structure_factor":
        if parser.posterior_distribution == "foldednormal":
            from abismal.surrogate_posterior.structure_factor.folded_normal import (
                FoldedNormalPosterior as Posterior,
            )
        elif parser.posterior_distribution == "truncatednormal":
            from abismal.surrogate_posterior.structure_factor.truncated_normal import (
                TruncatedNormalPosterior as Posterior,
            )
        elif parser.posterior_distribution == "rice":
            from abismal.surrogate_posterior.structure_factor.rice import (
                RicePosterior as Posterior,
            )
        elif parser.posterior_distribution == "nakagami":
            from abismal.surrogate_posterior.structure_factor.nakagami import (
                NakagamiPosterior as Posterior,
            )
        elif parser.posterior_distribution == "lognormal":
            from abismal.surrogate_posterior.structure_factor.lognormal import (
                LogNormalPosterior as Posterior,
            )
        elif parser.posterior_distribution == "normal":
            if parser.posterior_rank == 1:
                from abismal.surrogate_posterior.structure_factor.normal import (
                    NormalPosterior as Posterior,
                )
            else:
                from abismal.surrogate_posterior.structure_factor.normal import (
                    MultivariateNormalPosterior as Posterior,
                )

                posterior_kwargs["rank"] = parser.posterior_rank

    surrogate_posterior = Posterior(**posterior_kwargs)

    scale_model = ImageScaler(
        metadata_length=train.metadata_length,
        mlp_width=parser.d_model,
        mlp_depth=parser.layers,
        hidden_units=parser.d_model * 2,
        activation=parser.activation,
        kl_weight=parser.scale_kl_weight,
        epsilon=parser.epsilon,
        ff_epsilon=parser.ff_epsilon,
        prior_name=parser.scale_prior_distribution,
        posterior_name=parser.scale_posterior_distribution,
        bijector_name=parser.scale_posterior_bijector,
        pre_activation=parser.pre_activation,
        gated=parser.gated,
        optimize_prior_scale=parser.optimize_scale_prior,
        dropout=parser.dropout,
        batch_normalize=parser.batch_normalize,
        metadata_noise_factor=parser.metadata_noise_factor,
        share_weights=True,
    )

    if parser.studentt_dof is not None:
        likelihood = StudentTLikelihood(parser.studentt_dof)
    else:
        likelihood = NormalLikelihood()

    model = VariationalMergingModel(
        scale_model,
        surrogate_posterior,
        prior=prior,
        likelihood=likelihood,
        mc_samples=parser.mc_samples,
        kl_weight=parser.kl_weight,
        reindexing_ops=reindexing_ops,
        standardize=standardize_inputs,
    )

    if parser.learning_rate_final is not None:
        # The TF version built this schedule and then handed the optimizer the
        # constant rate anyway, so --learning-rate-final did nothing.
        learning_rate = piecewise_constant(
            parser.burnin, parser.learning_rate, parser.learning_rate_final
        )
    else:
        learning_rate = parser.learning_rate

    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "beta_1": parser.beta_1,
        "beta_2": parser.beta_2,
        "epsilon": parser.adam_epsilon,
        "clipnorm": parser.clipnorm,
        "clipvalue": parser.clip,
        "global_clipnorm": parser.global_clipnorm,
    }
    from abismal.optimizers.optimizer_dict import optimizer_dict

    # Passed only when set, so every optimizer keeps its own default -- notably
    # AdamW's 0.004, which an unconditional None would silently turn back into
    # plain Adam.
    if parser.weight_decay is not None:
        optimizer_kwargs['weight_decay'] = parser.weight_decay

    Optimizer = optimizer_dict[parser.optimizer]

    if parser.separate_friedel_mates:
        mtz_saver = FriedelMtzSaver(out_dir, dmin=dm.dmin)
    else:
        mtz_saver = MtzSaver(out_dir, parser.reference_mtz, dmin=dm.dmin)

    history_saver = HistorySaver(out_dir, gpu_id=parser.gpu_id, start_time=start_time)
    weight_saver = WeightSaver(out_dir)
    freezer = StandardizationFreezer()

    callbacks = [
        mtz_saver,
        history_saver,
        weight_saver,
        freezer,
    ]
    monitor = parser.early_stopping_criterion
    mode = 'min'
    if 'CC' in monitor:
        mode = 'max'
    if 'KL' in monitor:
        mode = 'max'
    early_stopping = EarlyStopping(monitor=monitor, patience=parser.early_stopping_patience, mode=mode)
    callbacks.append(early_stopping)

    if parser.eff_files is not None:
        for i, eff_file in enumerate(map(str, parser.eff_files)):
            pfx = f"eff_{i}"
            if parser.anomalous:
                phenix_runner = AnomalousPeakFinder(
                    out_dir,
                    eff_file,
                    epoch_stride=parser.phenix_frequency,
                    asu_id=0,
                    output_prefix=pfx,
                )
            elif parser.separate:
                phenix_runner = DifferenceMap(
                    out_dir,
                    eff_file,
                    epoch_stride=parser.phenix_frequency,
                    asu_id=0,
                    output_prefix=pfx,
                )
            else:
                phenix_runner = PhenixRunner(
                    out_dir,
                    eff_file,
                    epoch_stride=parser.phenix_frequency,
                    asu_id=0,
                    output_prefix=pfx,
                )
            callbacks.append(phenix_runner)

    if parser.torchref_pdb is not None:
        # The worker imports torchref in a detached subprocess whose stderr is a
        # file nobody reads, so a missing install does not stop anything -- it
        # produces a run that trains to completion and silently refines nothing.
        # Checked here, while there is still someone watching the terminal.
        #
        # find_spec rather than an import: torchref pulls torch, and torch in the
        # same interpreter as some TensorFlow versions segfaults. Asking whether
        # it could be imported is the whole question, so ask only that.
        from importlib.util import find_spec

        if find_spec("torchref") is None:
            raise SystemExit(
                "--torchref-pdb was given but the torchref package is not "
                "installed, so every refinement would fail in a subprocess whose "
                "output you would never see.\n"
                "    pip install 'abismal[torchref]'"
            )

        for i, pdb_file in enumerate(map(str, parser.torchref_pdb)):
            torchref_runner = TorchRefRunner(
                out_dir,
                pdb_file,
                epoch_stride=parser.torchref_frequency,
                asu_id=0,
                output_prefix=f"torchref_{i}",
                device='cpu',
                z_score_cutoff=parser.torchref_z_score,
                r_free_mtz=parser.r_free_mtz,
                r_free_value=parser.r_free_value,
                wavelength=parser.torchref_wavelength,
                adp_mode=parser.torchref_adp_mode,
                adp_aniso_sigma=parser.torchref_adp_aniso_sigma,
                rigid_body=not parser.torchref_no_rigid_body,
                rigid_body_iter=parser.torchref_rigid_body_iter,
                allow_overlap=parser.torchref_allow_overlap,
            )
            callbacks.append(torchref_runner)

    if parser.scale_init_file is not None:
        logger.info(f"Initializing the scale model from {parser.scale_init_file}")
        ref_model = load_model(str(parser.scale_init_file))
        model.scale_model.load_state_dict(ref_model.scale_model.state_dict())

    if parser.posterior_init_file is not None:
        logger.info(
            f"Initializing the surrogate posterior from {parser.posterior_init_file}"
        )
        ref_model = load_model(str(parser.posterior_init_file))
        model.surrogate_posterior.load_state_dict(
            ref_model.surrogate_posterior.state_dict()
        )

    if parser.freeze_scales:
        logger.info("Freezing the scale model")
        model.scale_model.requires_grad_(False)

    if parser.freeze_posterior:
        logger.info("Freezing the surrogate posterior")
        model.surrogate_posterior.requires_grad_(False)

    model.to(device)
    opt = Optimizer(parameter_groups(model), **optimizer_kwargs)

    step_fn = None
    if parser.compile:
        logger.info("Compiling model")
        step_fn = torch.compile(model, dynamic=True)

    trainer = Trainer(
        model,
        opt,
        device=device,
        callbacks=callbacks,
        verbose=parser.verbosity,
        step_fn=step_fn,
    )

    if parser.embed:
        logger.info("Embed selected, entering interactive, IPython shell.")
        from IPython import embed
        embed(colors='linux')


    logger.info("Starting training...")
    history = trainer.fit(
        train,
        epochs=parser.epochs,
        steps_per_epoch=dm.steps_per_epoch,
        validation_data=test,
        validation_steps=parser.validation_steps,
    )

    logger.info("Finished training.")

    if parser.embed:
        logger.info("Embed selected, entering interactive, IPython shell.")
        from IPython import embed

        embed(colors="linux")


if __name__ == "__main__":
    main()
