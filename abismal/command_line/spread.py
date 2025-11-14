"""
Estimate f', f'' for anomalous scatterers
"""



def main():
    import tf_keras as tfk
    from abismal.io.manager import DataManager
    from abismal.callbacks import MtzSaver
    from argparse import ArgumentParser
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--epochs", help="How many gradient descent epochs to run", type=int, default=30, required=False
    )
    parser.add_argument(
        "--steps-per-epoch", help="How many steps per epoch", type=int, default=None, required=False
    )
    parser.add_argument(
        "--batch-size", help="Number of images considered in each gradient step", type=int, default=100, required=False
    )
    parser.add_argument(
        "--keras-verbosity", help="Keras Model.fit verbose level. See docs for more info: https://keras.io/2.18/api/models/model_training_apis/#fit-method", type=int, default=1, choices=[0, 1, 2,]
    )
    parser.add_argument(
        "--dmin", type=float, default=None, help='Resolution cutoff for processing.',
    )
    parser.add_argument(
        "--model-file", type=str, required=True, help='A pdb file with the coordinates of anomalous scattering elements.',
    )
    parser.add_argument(
        "--fcalc-file", type=str, required=True, help='An mtz file containing the calculated structure factors for a model.',
    )
    parser.add_argument(
        "--data-files", type=str, nargs='+', required=True, help='The integrated diffraction data on which to conduct the "SPREAD" analysis.',
    )
    parser.add_argument(
        "--cell", type=float, nargs=6, help='Override the pdb cell constants.',
    )
    parser.add_argument(
        "--space-group", type=str, help='Override the pdb space group.',
    )
    parser = parser.parse_args()
    pdb = gemmi.read_pdb(parser.model_file)
    spacegroup = parser.space_group
    cell = parser.cell

    if spacegroup is None:
        spacegroup = pdb.spacegroup_hm
    if cell is None:
        cell = pdb.

    dm = DataManager(
        parser.data_files,
        dmin=parser.dmin,
        cell=cell,
        spacegroup=spacegroup,
    )
    train,test = dm.get_train_test_splits()
    train = train.batch(parser.batch_size)
    test = test.batch(parser.batch_size)

    surrogate_posterior = SpreadPosterior.from_files(
        model_file=parser.model_file,
        fcalc_file=parser.fcalc_file,
        cell = cell,
        spacegroup = spacegroup,
    )

    from abismal.merging.merging import VariationalMergingModel
    model = VariationalMergingModel(

    )
    opt = tfk.optimizers.Adam() #TODO add params
    for 

            model = tfk.saving.load_model(parser.model_file)
            if parser.sf_init is not None:
                sf_model = tfk.saving.load_model(parser.sf_init)
                model.surrogate_posterior.set_weights(
                    sf_model.surrogate_posterior.get_weights())

            model.trainable = False
            model.surrogate_posterior.trainable = True

            #Reset optimizer state to remove frozen variables
            opt = model.optimizer.from_config(model.optimizer.get_config()) 

            #Now re-compile to re-initialize the optimizer momenta
            model.compile(opt)

            callbacks = [
                MtzSaver(f"half_{half_id+1}", reference_mtz=parser.reference_mtz)
            ]
            history = model.fit(
                x=half, 
                epochs=parser.epochs, 
                steps_per_epoch=parser.steps_per_epoch, 
                callbacks=callbacks, 
                verbose=parser.keras_verbosity,
            )

            for asu_id,(asu,ds) in enumerate(zip(model.surrogate_posterior.rac, model.surrogate_posterior.to_datasets())):
                if asu.anomalous:
                    ds = ds.stack_anomalous().dropna()
                ds = ds.reset_index()
                #Make compatible with careless
                ds = ds.rename(columns={
                    'SIGF' : 'SigF',
                    'SIGI' : 'SigI',
                 })
                ds['repeat'] = repeat
                ds['asu_id'] = asu_id
                ds['half'] = half_id
                refls.append(ds)

    import reciprocalspaceship as rs
    refls = rs.concat(refls, check_isomorphous=False).infer_mtz_dtypes()
    refls.write_mtz("abismal_xval.mtz")
    for asu_id,ds in refls.groupby('asu_id'):
        ds.write_mtz(f"abismal_xval_{asu_id}.mtz")
