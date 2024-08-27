#!/usr/bin/env python

def main():
    from abismal.ragged import quiet
    from abismal.command_line.parser import parser
    parser = parser.parse_args()

    from abismal.io.tf_settings import set_log_level, set_gpu
    set_log_level(parser.tf_log_level)
    set_gpu(parser.gpu_id)
    run_abismal(parser)


# TODO: refactor this filetype control flow into abismal.io
_file_endings = {
    'refl' : ('.refl', '.pickle'),
    'expt' : ('.expt', '.json'),
    'stream' : ('.stream',),
}

def _is_file_type(s, endings):
    for ending in endings:
        if s.endswith(ending):
            return True
    return False

def _is_stream_file(s):
    return _is_file_type(s, _file_endings['stream'])

def _is_refl_file(s):
    return _is_file_type(s, _file_endings['refl'])

def _is_expt_file(s):
    return _is_file_type(s, _file_endings['expt'])

def _is_dials_file(s):
    return _is_refl_file(s) or _is_expt_file(s)

def run_abismal(parser):
    import tf_keras as tfk
    from abismal import __version__ as version
    from abismal.symmetry import ReciprocalASU,ReciprocalASUCollection
    from abismal.merging import VariationalMergingModel
    from abismal.callbacks import HistorySaver,MtzSaver,PhenixRunner,AnomalousPeakFinder
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

    log_file = parser.out_dir + "/abismal.log"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logger.info(f"Starting abismal, version {version}")
    logger.info("Running with the following options... ")
    for k,v in vars(parser).items():
        logger.info(f"{k} : {v}")
    logger.info(str(vars(parser)))

    cell = parser.cell
    space_group = parser.space_group
    asu_id = 0
    if all([_is_stream_file(f) for f in parser.inputs]):
        from abismal.io import StreamLoader
        data = None
        cell = parser.cell

        for stream_file in parser.inputs:
            loader = StreamLoader(
                stream_file, 
                cell=cell, 
                dmin=parser.dmin, 
                asu_id=asu_id, 
                wavelength=parser.wavelength,
            )
            if parser.separate:
                asu_id += 1
            else:
                asu_id = 1
            if cell is None:
                cell = loader.cell
            _data = loader.get_dataset(
                num_cpus=parser.num_cpus,
                logging_level=parser.ray_log_level,
            )
            if data is None:
                data = _data
            else:
                data = data.concatenate(_data)

    elif all([_is_dials_file(f) for f in parser.inputs]):
        from abismal.io import StillsLoader
        expt_files = [f for f in parser.inputs if _is_expt_file(f)]
        refl_files = [f for f in parser.inputs if _is_refl_file(f)]

        data = None
        if parser.separate:
            for expt,refl in zip(expt_files, refl_files):
                loader = StillsLoader([expt], [refl], space_group, cell, parser.dmin, asu_id)
                asu_id += 1
                _data = loader.get_dataset()
                if data is None:
                    data = _data
                else:
                    data = data.concatenate(_data)
        else:
            loader = StillsLoader(
                expt_files, refl_files, space_group, cell, parser.dmin, asu_id=asu_id
            )
            data = loader.get_dataset()
            asu_id += 1
        if cell is None:
            cell = loader.cell
    else:
        raise ValueError(
            "Couldn't determine input file type. "
            "DIALS reflection tables and CrystFEL streams are supported."
        )
    if space_group is None:
        if hasattr(loader, 'spacegroup'):
            space_group = loader.spacegroup
        else:
            space_group = 'P1'

    # Gemmification
    if not isinstance(cell, gemmi.UnitCell):
        cell = gemmi.UnitCell(*cell)
    if not isinstance(space_group, gemmi.SpaceGroup):
        space_group = gemmi.SpaceGroup(space_group)

    # Handle setting up the test fraction, shuffle buffer, batching, etc
    test = None
    if parser.test_fraction > 0.:
        train,test = split_dataset_train_test(data, parser.test_fraction)
        test  = test.cache().repeat().ragged_batch(parser.batch_size)
        test = test.prefetch(AUTOTUNE)
    else:
        train = data
    train = train.cache().repeat()
    if parser.shuffle_buffer_size > 0:
        train = train.shuffle(parser.shuffle_buffer_size)
    train = train.ragged_batch(parser.batch_size)

    rasu = []
    for i in range(asu_id):
        rasu.append(ReciprocalASU(
            cell,
            space_group,
            parser.dmin,
            anomalous=parser.anomalous,
        ))
    rac = ReciprocalASUCollection(*rasu)

    reindexing_ops = ['x,y,z']
    if not parser.disable_index_disambiguation:
        ops = gemmi.find_twin_laws(
            cell,
            space_group,
            3.0,
            False
        )
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops] 

    if parser.intensity_posterior:
        from abismal.surrogate_posterior.intensity import FoldedNormalPosterior
        surrogate_posterior = FoldedNormalPosterior(
            rac, 
            kl_weight=parser.kl_weight,
            epsilon=parser.epsilon,
            scale_factor=parser.init_scale,
        )
    else:
        prior = None
        if parser.parents is not None:
            from abismal.surrogate_posterior.structure_factor.wilson import MultiWilsonPrior
            prior = MultiWilsonPrior(
                rac, 
                parser.parents, 
                parser.prior_correlation, 
                parser.reindexing_ops,
            )
        surrogate_posterior = FoldedNormalPosterior(
            rac, 
            kl_weight=parser.kl_weight,
            epsilon=parser.epsilon,
            scale_factor=parser.init_scale,
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
        likelihood=likelihood,
        mc_samples=parser.mc_samples,
        reindexing_ops=reindexing_ops,
    )

    if parser.learning_rate_final is not None:
        from tf_keras.optimizers.schedules import PolynomialDecay
        learning_rate = PolynomialDecay(
            parser.learning_rate,
            parser.epochs * parser.steps_per_epoch,
            end_learning_rate=parser.learning_rate_final,
        )
    else:
        learning_rate = parser.learning_rate

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

    mtz_saver = MtzSaver(parser.out_dir, parser.anomalous)
    history_saver = HistorySaver(parser.out_dir, gpu_id=parser.gpu_id)
    weight_saver  = ModelCheckpoint(
        filepath=f'{parser.out_dir}/model.keras', verbose=1)

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

    train = train.prefetch(AUTOTUNE)

    if parser.debug:
        for x,y in train:
            break
        model([i[:3,:5] for i in x])

    model.compile(opt, run_eagerly=parser.run_eagerly, jit_compile=parser.jit_compile)

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

