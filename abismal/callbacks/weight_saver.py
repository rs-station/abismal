from abismal.callbacks.base import Callback


class WeightSaver(Callback):
    """Save the model after every epoch as `epoch_{n}.pt` under `prefix`."""
    def __init__(self, prefix):
        self.filepath = f'{prefix}/epoch_{{epoch}}.pt'

    def save(self, epoch):
        from abismal.saving import save_model
        save_model(self.model, self.filepath.format(epoch=epoch), optimizer=self.trainer.optimizer)

    def on_train_begin(self, logs=None):
        """Save the initial model, before a single gradient step.

        This exists so that a later `abismal.cchalf --sf-init` run can start
        from the same structure factor initialization the parent run did.
        Unlike the keras version, the whole model is built in its constructor,
        so epoch_0 carries every weight.
        """
        self.save(0)

    def on_epoch_end(self, epoch, logs=None):
        self.save(epoch + 1)
