"""
Estimate cchalf from abismal output.
"""

def freeze_all_but_posterior(model):
    """
    Configure `model` so that only the structure factors are refined.
    """
    model.requires_grad_(False)
    model.surrogate_posterior.requires_grad_(True)
    return model


def main(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "datamanager_yml", help="A .yml file from an abismal run", default="datamanager.yml"
    )
    parser.add_argument(
        "model_file", help="A .pt file from an abismal run. An abismal 1.x .keras file is converted on the fly.",
    )
    parser.add_argument(
        "--sf-init", help="An optional .pt (or abismal 1.x .keras) file from an abismal run from which to initialize the structure factors", default=None, required=False
    )
    parser.add_argument(
        "--repeats", help="Number of random repeats to conduct. Default is one.", type=int, default=1, required=False
    )
    parser.add_argument(
        "--verbosity", "--keras-verbosity", dest="verbosity", help="0 is silent, 1 shows a progress bar and 2 prints one line per epoch.", type=int, default=1, choices=[0, 1, 2,]
    )
    parser.add_argument(
        "--reference-mtz", type= str, default=None, help='A reference mtz file which will be used to determine the reindexing operator.',
    )
    parser.add_argument(
        "--run-eagerly", help="Accepted for compatibility with abismal 1.x and ignored.", action='store_true',
    )
    parser.add_argument(
        "--gpu-id", help="The numerical id of the GPU to use. Ignored when no GPU is available.", type=int, default=0,
    )
    parser.add_argument(
        "--device", help="Run on this PyTorch device, for instance 'cpu' or 'cuda:1'. Overrides --gpu-id.", type=str, default=None,
    )
    parser = parser.parse_args(args)

    from abismal.device import select_device, configure
    from abismal.io.manager import DataManager
    from abismal.callbacks import MtzSaver
    from abismal.saving import load_model, load_optimizer
    from abismal.training import Trainer

    device = select_device(parser.gpu_id, parser.device)
    configure(device)

    refls = []


    for repeat in range(parser.repeats):
        dm = DataManager.from_file(parser.datamanager_yml)
        half1,half2 = dm.get_half_datasets()

        for half_id,half in enumerate([half1, half2]):
            model = load_model(parser.model_file)
            if parser.sf_init is not None:
                sf_model = load_model(parser.sf_init)
                model.surrogate_posterior.load_state_dict(
                    sf_model.surrogate_posterior.state_dict())

            freeze_all_but_posterior(model)
            model.to(device)

            # Same optimizer and hyperparameters, but fresh moments, over the
            # posterior only.
            opt = load_optimizer(parser.model_file, model, state=False)
            if opt is None:
                from abismal.optimizers import Adam
                from abismal.training import parameter_groups
                opt = Adam(parameter_groups(model))

            callbacks = [
                MtzSaver(f"half_{half_id+1}", reference_mtz=parser.reference_mtz, dmin=dm.dmin)
            ]
            trainer = Trainer(model, opt, device=device, callbacks=callbacks, verbose=parser.verbosity)
            history = trainer.fit(
                half, 
                epochs=dm.epochs, 
                steps_per_epoch=dm.steps_per_epoch, 
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
