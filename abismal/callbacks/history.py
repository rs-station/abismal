import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import exists,dirname,abspath
from os import mkdir



class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, output_directory,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_directory = abspath(output_directory)

        if not exists(self.output_directory):
            mkdir(output_directory)


    def on_epoch_begin(self, epoch, logs=None):
        self.save_history()

    def on_train_end(self, epoch, logs=None):
        self.save_history()

    def save_history(self):
        out_file = f"{self.output_directory}/history.csv"
        df = pd.DataFrame(self.model.history.history)
        df['Epoch'] = np.arange(1, len(df)+1)
        df = df.set_index('Epoch', drop=True)
        df.to_csv(out_file)


