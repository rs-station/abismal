import tensorflow as tf
import tf_keras as tfk
import numpy as np
import pandas as pd
from os.path import exists,dirname,abspath
from os import mkdir
from time import time
from subprocess import run



class HistorySaver(tfk.callbacks.Callback):
    def __init__(self, output_directory, gpu_id=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id=0
        self.output_directory = abspath(output_directory)
        self.start = time()

        if not exists(self.output_directory):
            mkdir(output_directory)
        self.records = []

    def on_epoch_begin(self, epoch, logs=None):
        self.save_history(epoch)

    def on_train_end(self, epoch, logs=None):
        epoch = self.records[-1]['Epoch'] + 1
        self.save_history(epoch)

    def save_history(self, epoch):
        out_file = f"{self.output_directory}/history.csv"

        record = {k:v[-1] for k,v in self.model.history.history.items()}
        record['Epoch'] = epoch
        record['Time (s)'] = time() - self.start

        if self.gpu_id is not None:
            for k,v in self.get_gpu_memory_usage().items():
                record[k] = v
        self.records.append(record)

        df = pd.DataFrame.from_records(self.records)
        df.to_csv(out_file, index=False)

    def get_gpu_memory_usage(self):
        nvidia_smi_cmd = f"nvidia-smi -q -i {self.gpu_id} -d MEMORY"
        result = run(nvidia_smi_cmd.split(), capture_output=True)
        txt = result.stdout.decode()
        usage = {}

        bank = None
        for line in txt.split('\n'):
            if 'Memory Usage'  in line:
                bank = ' '.join(line.split()[0:-2])
                continue
            if bank is None:
                continue
            if line=='':
                continue
            stat = line.split()[0]
            unit = line.split()[-1]
            k = f'{bank} {stat} ({unit})'
            v = int(line.split()[-2])
            usage[k] = v

        return usage


