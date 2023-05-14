import reciprocalspaceship as rs
import tensorflow as tf
from os.path import exists,dirname,abspath
from os import mkdir

class MtzSaver(tf.keras.callbacks.Callback):
    def __init__(self, output_directory, anomalous=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_directory = abspath(output_directory)

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_epoch_end(self, epoch, logs=None):
        self.save_mtz(epoch)

    def save_mtz(self, epoch):
        for asu_id,posterior in enumerate(self.model.surrogate_posterior.posteriors):
            out = posterior.to_dataset(seen=True)
            out.write_mtz(f"{self.output_directory}/asu_{asu_id}_epoch_{epoch+1}.mtz")


