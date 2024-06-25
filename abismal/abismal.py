#!/usr/bin/env python

def main():
    from abismal.parser import parser
    parser = parser.parse_args()
    run_abismal(parser)

def run_abismal(parser):
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

    set_gpu(parser.gpu_id)
    cell = parser.cell
    if all([f.endswith('.stream') for f in parser.inputs]):
        from abismal.io import StreamLoader
        result = None
        cell = parser.cell
        for stream_file in parser.inputs:
            loader = StreamLoader(
                stream_file, 
                cell=cell, 
                dmin=parser.dmin, 
                asu_id=0, 
                wavelength=parser.wavelength,
            )
            if cell is None:
                cell = loader.cell
            data = loader.get_dataset()
            if result is None:
                result = data
            else:
                result = result.concatenate(data)
    elif all([f[-4:] in ('.expt', '.refl', '.json', '.pickle') for f in parser.inputs]):
        from abismal.io import StillsLoader
        expt_files = [i for i in parser.inputs if i in ('.expt', '.json')]
        refl_files = [i for i in parser.inputs if i in ('.refl', '.pickle')]
        loader = StillsLoader(parser)
        result = loader.get_dataset()
    else:
        raise ValueError(
            "Couldn't determine input file type. "
            "DIALS reflection tables and CrystFEL streams are supported."
        )

    # Gemmification
    cell = gemmi.UnitCell(*cell)
    space_group = gemmi.SpaceGroup(parser.space_group)

    # Handle setting up the test fraction, shuffle buffer, batching, etc
    if parser.test_fraction > 0.:
        train,test = train,test = split_dataset_train_test(data, parser.test_fraction)
        test  = test.cache().repeat().ragged_batch(parser.batch_size)
    train = train.cache().repeat()
    if parser.shuffle_buffer_size > 0:
        train = train.shuffle(parser.shuffle_buffer_size)
    train = train.ragged_batch(parser.batch_size)

    rasu = ReciprocalASU(
        cell,
        space_group,
        parser.dmin,
        anomalous=parser.anomalous,
    )
    rac = ReciprocalASUCollection(rasu)

    reindexing_ops = ['x,y,z']
    if not parser.disable_index_disambiguation:
        ops = gemmi.find_twin_laws(
            cell,
            space_group,
            3.0,
            False
        )
        reindexing_ops = reindexing_ops + [op.triplet() for op in ops] 

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
        eps=parser.epsilon,
        num_image_samples=parser.sample_reflections_per_image,
    )

    if parser.studentt_dof is not None:
        from abismal.likelihood import StudentTLikelihood
        likelihood = StudentTLikelihood(parser.studentt_dof)
    else:
        from abismal.likelihood import NormalLikelihood
        likelihood = NormalLikelihood()

    model = VariationalMergingModel(
        scale_model, 
        surrogate_posterior, 
        likelihood=likelihood,
        mc_samples=parser.mc_samples,
        reindexing_ops=reindexing_ops,
    )

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
    history_saver = HistorySaver(parser.out_dir)
    weight_saver  = ModelCheckpoint(
        filepath=f'{parser.out_dir}/abismal.weights.h5', save_weights_only=True, verbose=1)

    callbacks = [
        mtz_saver,
        history_saver,
        weight_saver,
    ]

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


    train,test = train.prefetch(AUTOTUNE),test.prefetch(AUTOTUNE)
    model.compile(opt, run_eagerly=parser.run_eagerly)
    history = model.fit(
        x=train, 
        epochs=parser.epochs, 
        steps_per_epoch=parser.steps_per_epoch, 
        validation_steps=parser.validation_steps, 
        callbacks=callbacks, 
        validation_data=test
    )

if __name__=='__main__':
    main()

