import tf_keras as tfk
import tensorflow as tf

class StandardizationFreezer(tfk.callbacks.Callback):
    """
    Freeze standardization layers after some epochs
    """
    def __init__(self, freeze_after=1, **kwargs):
        super().__init__(**kwargs)
        self.freeze_after = freeze_after

    def on_epoch_end(self, epoch, logs=None):
        """ Freeze standardization layers if necessary """
        if (epoch + 1) >= self.freeze_after:
            if self.model.standardize_intensity.trainable:
                tf.print("Freezing intensity standardization layer...")
                self.model.standardize_intensity.trainable = False
            if self.model.standardize_metadata.trainable:
                tf.print("Freezing metadata standardization layer...")
                self.model.standardize_metadata.trainable = False

