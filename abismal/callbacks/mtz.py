import tensorflow as tf
import tf_keras as tfk
from os.path import exists,dirname,abspath
import reciprocalspaceship as rs
from os import mkdir

class MtzSaver(tfk.callbacks.Callback):
    def __init__(self, output_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_directory = abspath(output_directory)

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_epoch_end(self, epoch, logs=None):
        self.save_mtz(epoch)

    def save_mtz(self, epoch, seen=True):
        for asu_id,data in enumerate(self.model.surrogate_posterior.to_datasets(seen=seen)):
            data.write_mtz(f"{self.output_directory}/asu_{asu_id}_epoch_{epoch+1}.mtz")

class FriedelMtzSaver(MtzSaver):
    """
    Save friedelized inputs into a single mtz.
    """
    def save_mtz(self, epoch, seen=True):
        ds_plus,ds_minus = self.model.surrogate_posterior.to_datasets(seen=seen)
        data = rs.concat((
            ds_plus,
            ds_minus.apply_symop('-x,-y,-z'),
        )).unstack_anomalous()
        data.write_mtz(f"{self.output_directory}/asu_0_epoch_{epoch+1}.mtz")


