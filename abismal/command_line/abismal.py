#!/usr/bin/env python

def main(args=None):
    from time import time
    start_time = time()
    from abismal.command_line.parser import parser

    parser = parser.parse_args(args)

    import torch
    from abismal.device import select_device, configure, resolve_compile, resolve_precision

    device = select_device(parser.gpu_id, parser.device)
    configure(device)
    precision = resolve_precision(parser.precision, device)
    compile_model = resolve_compile(parser.compile, device)
    if parser.seed is not None:
        torch.manual_seed(parser.seed)

    from abismal import __version__ as version
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
    from abismal.training import Trainer, EarlyStopping
    from abismal.saving import load_weights
    import logging
    from os.path import exists
    from os import mkdir

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

    logger.info(f"Training on {device}, compile={compile_model}, precision={precision}")
    logger.info("Configuring data input")
    from abismal.status import status
    status("Reading the data")
    dm = DataManager.from_parser(parser)
    train, test = dm.get_train_test_splits()
    if test is not None:
        logger.info("There is a test set for validation")

    from abismal.build import build_model, build_optimizer, facts_from
    status("Building the model")
    model = build_model(parser, facts_from(dm, train), data=train, precision=precision)

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
        # find_spec rather than an import: importing torchref is slow (numba) and
        # this process never uses it. Asking whether it could be imported is the
        # whole question, so ask only that.
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
                num_cpus=parser.torchref_num_cpus,
            )
            callbacks.append(torchref_runner)

    if parser.scale_init_file is not None:
        logger.info(f"Initializing the scale model from {parser.scale_init_file}")
        load_weights(str(parser.scale_init_file), model.scale_model, "scale_model")

    if parser.posterior_init_file is not None:
        logger.info(
            f"Initializing the surrogate posterior from {parser.posterior_init_file}"
        )
        load_weights(str(parser.posterior_init_file), model.surrogate_posterior, "surrogate_posterior")

    if parser.freeze_scales:
        logger.info("Freezing the scale model")
        model.scale_model.requires_grad_(False)

    if parser.freeze_posterior:
        logger.info("Freezing the surrogate posterior")
        model.surrogate_posterior.requires_grad_(False)

    model.to(device)
    opt = build_optimizer(parser, model)

    compile_options = None
    if compile_model:
        logger.info("Compiling model")
        compile_options = {"dynamic": True}

    trainer = Trainer(
        model,
        opt,
        device=device,
        callbacks=callbacks,
        verbose=parser.verbosity,
        compile=compile_options,
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
