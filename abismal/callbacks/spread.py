import tf_keras as tfk
import pandas as pd

class SpreadSaver(tfk.callbacks.Callback):
    def __init__(self, prefix, npoints=100, **kwargs):
        self.fmt = f'{prefix}/spread_epoch_{{epoch}}.csv'
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        results = self.model.surrogate_posterior.get_results()
        out = self.fmt.format(epoch=epoch + 1)
        results.to_csv(out)


