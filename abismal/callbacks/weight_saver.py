import warnings

import tf_keras as tfk


class WeightSaver(tfk.callbacks.ModelCheckpoint):
    def __init__(self, prefix, **kwargs):
        fstring = f'{prefix}/epoch_{{epoch}}.keras'
        super().__init__(filepath=fstring, **kwargs)

    def on_train_begin(self, logs):
        """Save the initial model, before a single gradient step.

        This exists so that a later `abismal.cchalf --sf-init` run can start
        from the same structure factor initialization the parent run did. The
        surrogate posterior is built in its constructor, so its weights are
        already there and epoch_0.keras carries all of them.

        The scale model is not, and neither are the standardization layers:
        those build lazily on first call, which has not happened yet. Keras
        warns about that -- "You are saving a model that has not yet been
        built. It might not contain any weights yet" -- and the warning is
        suppressed rather than fixed, deliberately.

        Building the model here would mean pulling a batch through it before
        training, which fills the shuffle buffer an extra time. That is one of
        the slowest steps in data loading and it would be paid twice, to write
        weights into a file whose only consumer wants the posterior. So the
        partial file is the intended artifact, and the warning is inaccurate
        about it: some weights are present, and they are the ones that matter.

        Narrowly scoped on purpose -- one message, one category, one call --
        so that a genuine unbuilt-model save anywhere else still warns.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are saving a model that has not yet been built",
                category=UserWarning,
            )
            tfk.saving.save_model(self.model, self.filepath.format(epoch=0))
