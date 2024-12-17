import math
import tf_keras as tfk

class WeightSaver(tfk.callbacks.ModelCheckpoint):
    def __init__(self, prefix, num_epochs, **kwargs):
        fsize = int(math.log10(num_epochs)) + 1
        fstring = f'{prefix}/epoch_{{epoch:0{fsize}d}}.keras'
        super().__init__(filepath=fstring, **kwargs)

    def on_train_begin(self, logs):
        """ Save initial model """
        tfk.saving.save_model(self.model, self.filepath.format(epoch=0))

