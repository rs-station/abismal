#!/usr/bin/env python


def main():
    from time import time
    start_time = time()
    from abismal.ragged import quiet
    from abismal.command_line.parser import parser

    parser = parser.parse_args()

    from abismal.io.tf_settings import set_log_level, set_gpu

    set_log_level(parser.tf_log_level)
    set_gpu(parser.gpu_id)
    run_abismal(parser, start_time)


def run_abismal(parser, start_time=None):
    import math
    import tensorflow as tf
    import tf_keras as tfk
    from abismal import __version__ as version
    from abismal.symmetry import (
        ReciprocalASU,
        ReciprocalASUCollection,
        ReciprocalASUGraph,
    )
    from abismal.merging import VariationalMergingModel
    from abismal.callbacks import (
        HistorySaver,
        MtzSaver,
        FriedelMtzSaver,
        PhenixRunner,
        AnomalousPeakFinder,
        WeightSaver,
        StandardizationFreezer,
    )
    from abismal.io import split_dataset_train_test, set_gpu
    from abismal.scaling import ImageScaler
    from abismal.surrogate_posterior.structure_factor import FoldedNormalPosterior
    from tf_keras.callbacks import ModelCheckpoint
    import gemmi
    import logging
    from os.path import exists
    from os import mkdir
    from abismal.likelihood import StudentTLikelihood
    from abismal.likelihood import NormalLikelihood

    if not exists(parser.out_dir):
        mkdir(parser.out_dir)

    from abismal.io.manager import DataManager

    log_file = parser.out_dir + "/abismal.log"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logger.info(f"Starting abismal, version {version}")
    msg = "\n".join([f"  {k}: {v}" for k, v in vars(parser).items()])
    logger.info("Running with the following options... ")
    logger.info(msg)

    logger.info("Configuring data input")
    dm = DataManager.from_parser(parser)
    train, test = dm.get_train_test_splits()
    dm_file = parser.out_dir + "/datamanager.yml"
    dm.to_file(dm_file)
    logger.info(f"Data manager config written to: {dm_file}")
    if test is not None:
        logger.info("There is a test set for validation")

    rasu = []
    anomalous = False if parser.separate_friedel_mates else parser.anomalous
    logger.info(f"Data are anomalous (True/False): {anomalous}")
    for i in range(dm.num_asus):
        logger.info(f"Adding asu ID: {i}")
        rasu.append(
            ReciprocalASU(
                dm.cell,
                dm.spacegroup,
                dm.dmin,
                anomalous=anomalous,
            )
        )

    logger.info("Combining reciprocal ASUs as collection")
    rac = ReciprocalASUGraph(
        *rasu,
        parents=parser.parents,
        reindexing_ops=parser.reindexing_ops,
    )

    reindexing_ops = ["x,y,z"]
    if not parser.disable_index_disambiguation:
        ops = gemmi.find_twin_laws(dm.cell, dm.spacegroup, 3.0, False)
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops]
        logger.info(f"Adding disambiguation operators: {reindexing_ops}")

    if parser.prior_distribution == "wilson":
        if parser.parents is not None:
            from abismal.prior.structure_factor.wilson import MultiWilsonPrior

            prior = MultiWilsonPrior(
                rac,
                parser.prior_correlation,
            )
        elif parser.posterior_type == "intensity":
            from abismal.prior.intensity.wilson import WilsonPrior

            prior = WilsonPrior(rac)
        else:
            from abismal.prior.structure_factor.wilson import WilsonPrior

            prior = WilsonPrior(rac)
        loc_init = prior.flat_distribution().mean()
        loc_init = tf.ones_like(loc_init) * tf.math.reduce_mean(loc_init)
        scale_init = parser.init_scale * loc_init
    elif parser.prior_distribution == "normal":
        from abismal.prior.normal import NormalPrior

        prior = NormalPrior(rac)
        loc_init = tf.ones_like(prior.flat_distribution().mean())
        scale_init = parser.init_scale * loc_init
        if parser.posterior_rank > 1:
            from abismal.prior.normal import MultivariateNormalPrior

            prior = MultivariateNormalPrior(rac)
    elif parser.prior_distribution == "halfnormal":
        from abismal.prior.normal import HalfNormalPrior
        prior = HalfNormalPrior(rac)
        loc_init = tf.ones_like(prior.flat_distribution().mean())
        scale_init = parser.init_scale * loc_init

    posterior_kwargs = {
        "rac": rac,
        "loc_init": loc_init,
        "scale_init": scale_init,
        "epsilon": parser.epsilon,
    }
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
        mlp_width=parser.d_model,
        mlp_depth=parser.layers,
        hidden_units=parser.d_model * 2,
        activation=parser.activation,
        kl_weight=parser.scale_kl_weight,
        epsilon=parser.epsilon,
        num_image_samples=parser.sample_reflections_per_image,
        prior_name=parser.scale_prior_distribution,
        posterior_name=parser.scale_posterior_distribution,
        bijector_name=parser.scale_posterior_bijector,
        normalizer_name=parser.normalizer,
        gated=parser.gated,
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
        standardization_decay=parser.standardization_decay,
    )

    if parser.learning_rate_final is not None:
        from tf_keras.optimizers.schedules import PiecewiseConstantDecay
        boundaries = [parser.burnin]
        values = [parser.learning_rate, parser.learning_rate_final]
        learning_rate = PiecewiseConstantDecay(boundaries, values)
    else:
        learning_rate = parser.learning_rate

    optimizer_kwargs = {
        "learning_rate": parser.learning_rate,
        "beta_1": parser.beta_1,
        "beta_2": parser.beta_2,
        "epsilon": parser.adam_epsilon,
        "clipnorm": parser.clipnorm,
        "clipvalue": parser.clip,
        "global_clipnorm": parser.global_clipnorm,
        "lazy_vars" : [v._unique_id for v in surrogate_posterior.trainable_variables],
    }
    from abismal.optimizers.optimizer_dict import optimizer_dict

    Optimizer = optimizer_dict[parser.optimizer]
    opt = Optimizer(**optimizer_kwargs)

    if parser.separate_friedel_mates:
        mtz_saver = FriedelMtzSaver(parser.out_dir)
    else:
        mtz_saver = MtzSaver(parser.out_dir, parser.reference_mtz)

    history_saver = HistorySaver(parser.out_dir, gpu_id=parser.gpu_id, start_time=start_time)
    weight_saver = WeightSaver(parser.out_dir)
    freezer = StandardizationFreezer()

    callbacks = [
        mtz_saver,
        history_saver,
        weight_saver,
        freezer,
    ]

    if parser.eff_files is not None:
        for i, eff_file in enumerate(parser.eff_files.split(",")):
            pfx = f"eff_{i}"
            if parser.anomalous:
                f = AnomalousPeakFinder(
                    parser.out_dir,
                    eff_file,
                    epoch_stride=parser.phenix_frequency,
                    asu_id=0,
                    output_prefix=pfx,
                )
            else:
                f = PhenixRunner(
                    parser.out_dir,
                    eff_file,
                    epoch_stride=parser.phenix_frequency,
                    asu_id=0,
                    output_prefix=pfx,
                )
            callbacks.append(f)

    need_to_build = False
    need_to_build |= parser.debug
    need_to_build |= parser.scale_init_file is not None
    need_to_build |= parser.posterior_init_file is not None
    if need_to_build:
        logger.info(f"Initializing weights")
        for x, y in train:
            model(x)
            break

    if parser.scale_init_file is not None:
        logger.info(f"Initializing the scale model from {parser.scale_init_file}")
        ref_model = tfk.saving.load_model(parser.scale_init_file)
        model.scale_model.set_weights(ref_model.scale_model.get_weights())

    if parser.posterior_init_file is not None:
        logger.info(
            f"Initializing the surrogate posterior from {parser.posterior_init_file}"
        )
        ref_model = tfk.saving.load_model(parser.posterior_init_file)
        model.surrogate_posterior.set_weights(
            ref_model.surrogate_posterior.get_weights()
        )

    if parser.freeze_scales:
        logger.info("Freezing the scale model")
        model.scale_model.trainable = False

    if parser.freeze_posterior:
        logger.info("Freezing the surrogate posterior")
        model.surrogate_posterior.trainable = False

    logger.info("Compiling model")
    model.compile(opt, run_eagerly=parser.run_eagerly, jit_compile=parser.jit_compile)
    if parser.debug:
        from IPython import embed
        embed(colors='linux')

    # for x,y in train:
    #    model(x)
    #    break
    # with tf.GradientTape(persistent=True) as tape:
    #    y_pred = model(x, training=True)  # Forward pass
    #    # Compute the loss value
    #    # (the loss function is configured in `compile()`)
    #    loss = model.compiled_loss(y, y_pred, regularization_losses=model.losses)
    # q_vars = model.surrogate_posterior.trainable_variables
    # grad_q= tape.gradient(loss, q_vars)
    # from abismal.merging.merging import to_indexed_slices
    # gis_q = [to_indexed_slices(g) for g in grad_q]
    # from IPython import embed
    # embed(colors='linux')

    logger.info("Starting training...")
    history = model.fit(
        x=train,
        epochs=parser.epochs,
        steps_per_epoch=parser.steps_per_epoch,
        validation_steps=parser.validation_steps,
        callbacks=callbacks,
        validation_data=test,
        verbose=parser.keras_verbosity,
    )

    logger.info("Finished training.")

    if parser.debug:
        logger.info("Debug mode selected, entering interactive, IPython shell.")
        from IPython import embed

        embed(colors="linux")


if __name__ == "__main__":
    main()
