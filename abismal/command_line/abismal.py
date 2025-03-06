#!/usr/bin/env python

def main():
    from abismal.ragged import quiet
    from abismal.command_line.parser import parser
    parser = parser.parse_args()

    from abismal.io.tf_settings import set_log_level, set_gpu
    set_log_level(parser.tf_log_level)
    set_gpu(parser.gpu_id)
    run_abismal(parser)

def run_abismal(parser):
    import math
    import tensorflow as tf
    import tf_keras as tfk
    from abismal import __version__ as version
    from abismal.symmetry import ReciprocalASU,ReciprocalASUCollection,ReciprocalASUGraph
    from abismal.merging import VariationalMergingModel
    from abismal.callbacks import HistorySaver,MtzSaver,FriedelMtzSaver,PhenixRunner,AnomalousPeakFinder,WeightSaver
    from abismal.io import split_dataset_train_test,set_gpu
    from abismal.scaling import ImageScaler
    from abismal.surrogate_posterior.structure_factor import FoldedNormalPosterior
    from tf_keras.optimizers import Adam
    from tf_keras.callbacks import ModelCheckpoint
    import gemmi
    from tensorflow.data import AUTOTUNE
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
    logger.info("Running with the following options... ")
    msg = '\n'.join([f"{k}: {v}" for k,v in vars(parser).items()])
    logger.info(f"Run parameters: \n" + msg)
    logger.info(str(vars(parser)))

    dm = DataManager.from_parser(parser)
    train,test = dm.get_train_test_splits()
    dm.to_file(parser.out_dir + "/datamanager.yml")

    if test is not None:
        test  = test.cache().repeat().ragged_batch(parser.batch_size)
        test = test.prefetch(AUTOTUNE)
    train = train.cache().repeat()
    if parser.shuffle_buffer_size > 0:
        train = train.shuffle(parser.shuffle_buffer_size)
    train = train.ragged_batch(parser.batch_size)

    rasu = []
    anomalous = False if parser.separate_friedel_mates else parser.anomalous
    for i in range(dm.num_asus):
        rasu.append(ReciprocalASU(
            dm.cell,
            dm.spacegroup,
            dm.dmin,
            anomalous=anomalous,
        ))

    rac = ReciprocalASUGraph(
        *rasu,
        parents=parser.parents,
        reindexing_ops=parser.reindexing_ops,
    )

    reindexing_ops = ['x,y,z']
    if not parser.disable_index_disambiguation:
        ops = gemmi.find_twin_laws(
            dm.cell,
            dm.spacegroup,
            3.0,
            False
        )
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops] 

    if parser.parents is not None:
        from abismal.prior.structure_factor.wilson import MultiWilsonPrior
        from abismal.surrogate_posterior.structure_factor.folded_normal import FoldedNormalPosterior
        prior = MultiWilsonPrior(
            rac, 
            parser.prior_correlation, 
        )
        loc_init = prior.distribution(rac.asu_id[:,None], rac.Hunique).mean()
        scale_init = parser.init_scale * loc_init
        surrogate_posterior = FoldedNormalPosterior(
            rac, 
            loc_init,
            scale_init,
            epsilon=parser.epsilon,
        )
    elif parser.intensity_posterior:
        from abismal.surrogate_posterior.intensity import FoldedNormalPosterior
        from abismal.prior.intensity.wilson import WilsonPrior
        prior = WilsonPrior(rac)
        loc_init = prior.distribution(rac.asu_id[:,None], rac.Hunique).mean()
        scale_init = parser.init_scale * loc_init
        surrogate_posterior = FoldedNormalPosterior(
            rac, 
            loc_init,
            scale_init,
            epsilon=parser.epsilon,
        )
    else:
        from abismal.surrogate_posterior.structure_factor import FoldedNormalPosterior as Posterior
        #from abismal.surrogate_posterior.structure_factor.rice import RicePosterior as Posterior
        from abismal.prior.structure_factor.wilson import WilsonPrior
        prior = WilsonPrior(rac)
        loc_init = prior.distribution(rac.asu_id[:,None], rac.Hunique).mean()
        scale_init = parser.init_scale * loc_init
        surrogate_posterior = Posterior(
            rac, 
            loc_init,
            scale_init,
            epsilon=parser.epsilon,
        )

    scale_model = ImageScaler(
        mlp_width=parser.d_model, 
        mlp_depth=parser.layers, 
        hidden_units=parser.d_model * 2,
        activation=parser.activation,
        kl_weight=parser.scale_kl_weight,
        epsilon=parser.epsilon,
        num_image_samples=parser.sample_reflections_per_image,
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
    )

    if parser.learning_rate_final is not None:
        from tf_keras.optimizers.schedules import PiecewiseConstantDecay
        steps = parser.steps_per_epoch * parser.epochs
        boundaries = [ steps // 2 ]
        values = [ parser.learning_rate, parser.learning_rate_final ]
        learning_rate = PiecewiseConstantDecay(boundaries, values)
        #from tf_keras.optimizers.schedules import PolynomialDecay
        #learning_rate = PolynomialDecay(
        #    parser.learning_rate,
        #    parser.epochs * parser.steps_per_epoch,
        #    end_learning_rate=parser.learning_rate_final,
        #)
    else:
        learning_rate = parser.learning_rate


    if parser.use_wadam:
        from abismal.optimizers.wadam import WAdam
        opt = WAdam(
            parser.learning_rate, 
            parser.beta_1, 
            parser.beta_2, 
            global_clipnorm=parser.global_clipnorm, 
            clipnorm=parser.clipnorm, 
            clipvalue=parser.clip, 
            epsilon=parser.adam_epsilon, 
        )
    else:
        opt = Adam(
            parser.learning_rate, 
            parser.beta_1, 
            parser.beta_2, 
            global_clipnorm=parser.global_clipnorm, 
            clipnorm=parser.clipnorm, 
            clipvalue=parser.clip, 
            epsilon=parser.adam_epsilon, 
            amsgrad=parser.amsgrad
        )

    if parser.separate_friedel_mates:
        mtz_saver = FriedelMtzSaver(parser.out_dir)
    else:
        mtz_saver = MtzSaver(parser.out_dir)
    history_saver = HistorySaver(parser.out_dir, gpu_id=parser.gpu_id)
    weight_saver  = WeightSaver(parser.out_dir)

    callbacks = [
        mtz_saver,
        history_saver,
        weight_saver,
    ]

    if parser.eff_files is not None:
        for i,eff_file in enumerate(parser.eff_files.split(',')):
            pfx = f"eff_{i}"
            if parser.anomalous:
                f = AnomalousPeakFinder(
                    parser.out_dir, eff_file, epoch_stride=parser.phenix_frequency, 
                    asu_id=0, output_prefix=pfx
                )
            else:
                f = PhenixRunner(
                    parser.out_dir, eff_file, epoch_stride=parser.phenix_frequency, 
                    asu_id=0, output_prefix=pfx
                )
            callbacks.append(f)


    if parser.debug:
        for x,y in train:
            break
        model([i[:3,:5] for i in x])

    model.compile(opt, run_eagerly=parser.run_eagerly, jit_compile=parser.jit_compile)

    train = train.prefetch(AUTOTUNE)
    history = model.fit(
        x=train, 
        epochs=parser.epochs, 
        steps_per_epoch=parser.steps_per_epoch, 
        validation_steps=parser.validation_steps, 
        callbacks=callbacks, 
        validation_data=test
    )

    if parser.debug:
        from IPython import embed
        embed(colors='linux')

if __name__=='__main__':
    main()

