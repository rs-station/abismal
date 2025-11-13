"""
Estimate cchalf from abismal output.
"""



def main():
    import tf_keras as tfk
    from abismal.io.manager import DataManager
    from abismal.callbacks import MtzSaver
    from argparse import ArgumentParser
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "datamanager_yml", help="A .yml file from an abismal run", default="datamanager.yml"
    )
    parser.add_argument(
        "model_file", help="A .keras file from an abismal run",
    )
    parser.add_argument(
        "--sf-init", help="An optional .keras file from which to initialize the structure factors file from an abismal run", default=None, required=False
    )
    parser.add_argument(
        "--repeats", help="Number of random repeats to conduct. Default is one.", type=int, default=1, required=False
    )
    parser.add_argument(
        "--keras-verbosity", help="Keras Model.fit verbose level. See docs for more info: https://keras.io/2.18/api/models/model_training_apis/#fit-method", type=int, default=1, choices=[0, 1, 2,]
    )
    parser.add_argument(
        "--reference-mtz", type= str, default=None, help='A reference mtz file which will be used to determine the reindexing operator.',
    )
    parser = parser.parse_args()
    refls = []


    for repeat in range(parser.repeats):
        dm = DataManager.from_file(parser.datamanager_yml)
        half1,half2 = dm.get_half_datasets()

        for half_id,half in enumerate([half1, half2]):
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
                epochs=dm.epochs, 
                steps_per_epoch=dm.steps_per_epoch, 
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
