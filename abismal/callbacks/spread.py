import tf_keras as tfk
import pandas as pd

class SpreadSaver(tfk.callbacks.Callback):
    def __init__(self, prefix, npoints=100, **kwargs):
        self.csv_file = f'{prefix}/spread.csv'
        self.first_write = True
        super().__init__(**kwargs)

    def write_epoch_results(self, epoch):
        results = self.model.surrogate_posterior.get_results()
        results['Epoch'] = epoch
        results.to_csv(
            self.csv_file,
            mode='w' if self.first_write else 'a',
            header = self.first_write,
        )
        self.first_write = False

    def on_train_begin(self, logs):
        self.write_epoch_results(0)

    def on_epoch_end(self, epoch, logs):
        self.write_epoch_results(epoch + 1)



