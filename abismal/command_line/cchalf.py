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
        "structure_factor_model", help="An optional .keras file from which to initialize the structure factors file from an abismal run", default=None,
    )
    parser.add_argument(
        "--epochs", help="How many gradient descent epochs to run", type=int, default=30,
    )
    parser.add_argument(
        "--steps-per-epoch", help="How many steps per epoch", type=int, default=1_000,
    )
    parser.add_argument(
        "--batch-size", help="Number of images considered in each gradient step", type=int, default=100,
    )
    parser = parser.parse_args()


    dm = DataManager.from_file(parser.datamanager_yml)
    dm.test_fraction = 0.5
    half1,half2 = dm.get_train_test_splits()

    for i,half in enumerate([half1, half2]):
        half = half.cache().repeat().ragged_batch(parser.batch_size)
        model = tfk.saving.load_model(parser.model_file)
        if parser.structure_factor_model is not None:
            sf_model = tfk.saving.load_model(parser.structure_factor_model)
            model.surrogate_posterior.set_weights(
                sf_model.surrogate_posterior.get_weights())

        model.standardize_intensity.trainable = False
        model.standardize_metadata.trainable = False
        model.likelihood.trainable = False
        model.scale_model.trainable = False
        model.prior.trainable = False

        #Need to update this to 
        model.compile(model.optimizer)

        callbacks = [
            MtzSaver(f"half_{i+1}"),
        ]
        history = model.fit(
            x=half, 
            epochs=parser.epochs, 
            steps_per_epoch=parser.steps_per_epoch, 
            callbacks=callbacks, 
        )

        ref = tfk.saving.load_model(parser.model_file)
        import numpy as np
        unsame = []
        for w1,w2 in zip(ref.weights, model.weights):
            if w1.dtype in ('float32', 'float64'):
                same = np.allclose(w1, w2)
                if not same:
                    unsame.append(w1)


        from IPython import embed
        embed(colors='linux')
        XX

    from IPython import embed
    embed(colors='linux')
