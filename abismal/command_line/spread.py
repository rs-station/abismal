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
        SpreadSaver,
    )
    from abismal.prior.spread.spread import SpreadPrior

    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--epochs", help="How many gradient descent epochs to run", type=int, default=30, required=False
    )
    parser.add_argument(
        "--shuffle-buffer-size", help="Buff the shuffles", type=int, default=10_000, required=False
    )
    parser.add_argument(
        "--num-cpus", help="CPUs to use for data loading", type=int, default=1, required=False
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
        "--steps-per-epoch", help="How many steps per epoch", type=int, default=1_000, required=False
    )
    parser.add_argument(
        "--batch-size", help="Number of images considered in each gradient step", type=int, default=100, required=False
    )
    parser.add_argument(
        "--test-fraction", help="Fraction of data reserved for validation", type=float, default=0., required=False
    )
    parser.add_argument(
        "--scale-kl-weight", help="Scale KL divergence weight", type=float, default=1., required=False
    )
    parser.add_argument(
        "--kl-weight", help="KL divergence weight", type=float, default=1., required=False
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
        "--debug", action='store_true', help='Debug mode runs eagerly.',
    )
    parser.add_argument(
        "--unity-posterior", action='store_true', help='Use FCalc as the posterior.',
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
            parser.wavelength_range = SpreadPosterior.estimate_wavelength_range(parser.integrated, num_cpus=parser.num_cpus)

    if parser.unity_posterior:
        from abismal.surrogate_posterior.spread.spread import DummySpreadPosterior as SpreadPosterior
    surrogate_posterior = SpreadPosterior.from_pdb(
        pdb_file=parser.model_file,
        elements=parser.elements,
        dmin=parser.dmin,
        wavelength_range=parser.wavelength_range,
        energy_range=parser.energy_range,
        epsilon=1e-5,
    )
    prior = SpreadPrior.from_spread_posterior(surrogate_posterior)

    dm = DataManager(
        parser.integrated,
        dmin=parser.dmin,
        batch_size=parser.batch_size,
        cell=surrogate_posterior.cell,
        spacegroup=surrogate_posterior.spacegroup,
        test_fraction=parser.test_fraction,
        num_cpus=parser.num_cpus,
        steps_per_epoch=parser.steps_per_epoch,
        shuffle_buffer_size=parser.shuffle_buffer_size,
    )
    train,test = dm.get_train_test_splits()


    scale_model = ImageScaler(
            mlp_width=parser.d_model,
            mlp_depth=parser.layers,
            hidden_units=None,
            activation="relu",
            kl_weight=parser.scale_kl_weight,
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
        kl_weight=parser.kl_weight,
        reindexing_ops=reindexing_ops,
        standardization_decay=0.999,
    )

    #mtz_saver = MtzSaver(parser.out_dir, parser.reference_mtz)
    spread_saver = SpreadSaver(parser.out_dir)
    history_saver = HistorySaver(parser.out_dir, gpu_id=0, start_time=start_time)
    weight_saver = WeightSaver(parser.out_dir)
    freezer = StandardizationFreezer()

    callbacks = [
        #mtz_saver,
        history_saver,
        weight_saver,
        freezer,
        spread_saver,
    ]

    if not exists(parser.out_dir):
        mkdir(parser.out_dir)

    from abismal.optimizers import Adam
    opt = Adam()
    model.compile(opt, run_eagerly=parser.debug)

    for x,y in train:
        z = model(x)
        model.surrogate_posterior.get_results()
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

def plot_results():
    from argparse import ArgumentParser
    parser = ArgumentParser("Plot results from the spread training.")
    parser.add_argument(
        "csv", help="A 'spread_epoch_#.csv' file to plot.", type=str
    )
    parser.add_argument(
        "--epoch", help="Which epoch to plot. By default the most recent.", type=int, default=None
    )
    parser = parser.parse_args()

    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    import reciprocalspaceship as rs
    f,(ax1,ax2) = plt.subplots(2)

    results = pd.read_csv(parser.csv)

    epoch = None
    if 'Epoch' in results:
        epoch = parser.epoch
        if epoch is None:
            epoch = results['Epoch'].max()
        results = results[results.Epoch == epoch]
        del results['Epoch']

    results['energy'] = rs.utils.angstroms2ev(results['wavelength'])
    alpha = 0.2
    for atom,df in results.groupby("atom_name"):
        x = df['energy']
        y = df["f'"]
        s = df['stddev']

        fb = ax1.fill_between(
            x, y - s, y + s, alpha=alpha
        )
        c = fb.properties()['facecolor'][:3]
        ax1.plot(x, y, color=c, label=atom, alpha=1.0)
        y = df["f''"]
        ax2.fill_between(
            x, y - s, y + s, alpha=alpha, color=c
        )
        ax2.plot(x, y, color=c, alpha=1.0)


    if epoch is not None:
        plt.suptitle(f"Spread Results, Epoch {epoch}")
    ax1.set_ylabel("f' (arbtrary units)")
    ax2.set_ylabel("f'' (arbtrary units)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.show()

