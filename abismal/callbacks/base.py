class Callback:
    """
    Hooks into `abismal.training.Trainer.fit`, with the signatures of the
    keras callbacks abismal used before.

    The trainer sets `model` (the VariationalMergingModel) and `trainer` (for
    its `history`, a dict of per-epoch metric lists) before training starts.
    """
    model = None
    trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.model = trainer.model

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass
