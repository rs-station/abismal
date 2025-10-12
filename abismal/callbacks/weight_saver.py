import tf_keras as tfk

class WeightSaver(tfk.callbacks.ModelCheckpoint):
    def __init__(self, prefix, **kwargs):
        fstring = f'{prefix}/epoch_{{epoch}}.keras'
        super().__init__(filepath=fstring, **kwargs)

    def on_train_begin(self, logs):
        """ Save initial model """
        tfk.saving.save_model(self.model, self.filepath.format(epoch=0))

