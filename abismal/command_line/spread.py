"""
Estimate f', f'' for anomalous scatterers. 
"""



def main():
    from time import time
    start_time = time()
    from os.path import exists
    from os import mkdir
    import tf_keras as tfk
    import gemmi
    from abismal.io.manager import DataManager
    from abismal.callbacks import MtzSaver
    from argparse import ArgumentParser
    from abismal.merging.merging import SpreadMergingModel
    from abismal.likelihood import StudentTLikelihood
    from abismal.likelihood import NormalLikelihood
    from abismal.scaling import ImageScaler
    from abismal.callbacks import (
        HistorySaver,
        MtzSaver,
        FriedelMtzSaver,
        PhenixRunner,
        AnomalousPeakFinder,
        WeightSaver,
        StandardizationFreezer,
    )
    from abismal.prior.spread.spread import SpreadPrior

    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--epochs", help="How many gradient descent epochs to run", type=int, default=30, required=False
    )
    parser.add_argument(
        "--mc-samples", help="Number of mc samples used in gradient estimation", type=int, default=32, required=False
    )
    parser.add_argument(
        "--d-model", help="The width of the neural network", type=int, default=32, required=False
    )
    parser.add_argument(
        "--layers", help="The depth of the neural network", type=int, default=20, required=False
    )
    parser.add_argument(
        "--steps-per-epoch", help="How many steps per epoch", type=int, default=None, required=False
    )
    parser.add_argument(
        "--batch-size", help="Number of images considered in each gradient step", type=int, default=100, required=False
    )
    parser.add_argument(
        "--test-fraction", help="Fraction of data reserved for validation", type=float, default=0., required=False
    )
    parser.add_argument(
        "--studentt-dof", help="Student's t degrees of freedom for likelihood", type=float, default=None, required=False
    )
    parser.add_argument(
        "--keras-verbosity", help="Keras Model.fit verbose level. See docs for more info: https://keras.io/2.18/api/models/model_training_apis/#fit-method", type=int, default=1, choices=[0, 1, 2,]
    )
    parser.add_argument(
        "--dmin", required=True, type=float, default=None, help='Resolution cutoff for processing.',
    )
    parser.add_argument(
        "--model-file", type=str, required=True, help='A pdb files of the structure.',
    )
    parser.add_argument(
        "--out-dir", type=str, default='./', help='Where to save the output.',
    )
    parser.add_argument(
        "--elements", type=lambda x: x.split(','), required=True, help="List of elements for which to refine f' and f''. These should be specified as a comma-separated string ie. 'Mn,I,Fe,S'",
    )
    parser.add_argument(
        "--wavelength-range", default=None, type=float, nargs=2, help="Specify the wavelength range over which to refine f' and f''.",
    )
    parser.add_argument(
        "--energy-range", default=None, type=float, nargs=2, help="Specify the energy range over which to refine f' and f''. ",
    )
    parser.add_argument(
        "integrated", type=str, nargs='+', help='The integrated diffraction data on which to conduct the "SPREAD" analysis.',
    )
    parser = parser.parse_args()
    pdb = gemmi.read_pdb(parser.model_file)
    from abismal.surrogate_posterior.spread.spread import SpreadPosterior
    if parser.wavelength_range is None:
        if parser.energy_range is None:
            wavs = []
            parser.wavelength_range = SpreadPosterior.estimate_wavelength_range(parser.integrated)

    surrogate_posterior = SpreadPosterior.from_pdb(
        pdb_file=parser.model_file,
        elements=parser.elements,
        dmin=parser.dmin,
        wavelength_range=parser.wavelength_range,
        energy_range=parser.energy_range,
    )
    prior = SpreadPrior.from_spread_posterior(surrogate_posterior)

    dm = DataManager(
        parser.integrated,
        dmin=parser.dmin,
        batch_size=parser.batch_size,
        cell=surrogate_posterior.cell,
        spacegroup=surrogate_posterior.spacegroup,
        test_fraction=parser.test_fraction,
        num_cpus=1,
        steps_per_epoch=parser.steps_per_epoch,
        #shuffle_buffer_size=10_000,
    )
    train,test = dm.get_train_test_splits()


    scale_model = ImageScaler(
            mlp_width=parser.d_model,
            mlp_depth=parser.layers,
            hidden_units=None,
            activation="relu",
            kl_weight=1.,
            epsilon=1e-12,
            num_image_samples=None,
            share_weights=True,
            prior_name='lognormal',
            posterior_name='foldednormal',
            bijector_name='softplus',
            normalizer_name='rms',
            hkl_to_imodel=False,
            gated=False,
            output_bias=True,
    )
    if parser.studentt_dof is not None:
        likelihood = StudentTLikelihood(parser.studentt_dof)
    else:
        likelihood = NormalLikelihood()

    reindexing_ops = ["x,y,z"]
    ops = gemmi.find_twin_laws(dm.cell, dm.spacegroup, 3.0, False)
    reindexing_ops = reindexing_ops + [op.triplet() for op in ops]

    model = SpreadMergingModel(
        scale_model,
        surrogate_posterior,
        prior=prior,
        likelihood=likelihood,
        mc_samples=parser.mc_samples,
        kl_weight=1.,
        reindexing_ops=reindexing_ops,
        standardization_decay=0.999,
    )

    #mtz_saver = MtzSaver(parser.out_dir, parser.reference_mtz)
    history_saver = HistorySaver(parser.out_dir, gpu_id=0, start_time=start_time)
    weight_saver = WeightSaver(parser.out_dir)
    freezer = StandardizationFreezer()

    callbacks = [
        #mtz_saver,
        history_saver,
        #weight_saver,
        freezer,
    ]

    if not exists(parser.out_dir):
        mkdir(parser.out_dir)

    from abismal.optimizers import Adam
    opt = Adam()
    model.compile(opt, run_eagerly=True)

    for x,y in train:
        z = model(x)
        break

    history = model.fit(
        x=train,
        epochs=parser.epochs,
        steps_per_epoch=parser.steps_per_epoch,
        #validation_steps=parser.validation_steps,
        callbacks=callbacks,
        validation_data=test,
        verbose=parser.keras_verbosity,
    )


