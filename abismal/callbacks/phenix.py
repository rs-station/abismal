import reciprocalspaceship as rs
import tensorflow as tf
from os.path import exists,dirname,abspath
from os import mkdir
from subprocess import Popen,DEVNULL
from abismal.callbacks import MtzSaver

class PhenixRunner(tf.keras.callbacks.Callback):
    """ A hybrid callback that saves mtz files and optionally runs PHENIX """
    def __init__(self, output_directory, eff_file=None, epoch_stride=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_directory = abspath(output_directory)
        self.eff_file = eff_file
        self.epoch_stride = epoch_stride

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_epoch_end(self, epoch, logs=None):
        if self.eff_file is not None and (epoch + 1) % self.epoch_stride == 0:
            self.save_mtz(epoch, True)
        else:
            self.save_mtz(epoch, False)

    def save_mtz(self, epoch, run_phenix=False, seen=True):
        for asu_id,data in enumerate(self.model.surrogate_posterior.to_datasets(seen=seen)):
            mtz_file = f"{self.output_directory}/asu_{asu_id}_epoch_{epoch+1}.mtz"
            data.write_mtz(f"{self.output_directory}/asu_{asu_id}_epoch_{epoch+1}.mtz")
            if run_phenix:
                phenix_dir = f"{self.output_directory}/phenix_asu_{asu_id}_epoch_{epoch+1}"
                if not exists(phenix_dir):
                    mkdir(phenix_dir)
                command = [
                    'phenix.refine',
                    self.eff_file,
                    mtz_file,
                ]

                stderr = phenix_dir + '/stderr.txt'
                stdout = phenix_dir + '/stdout.txt'
                with open(stderr, 'w') as e, open(stdout, 'w') as o:
                    p = Popen(
                        command, 
                        cwd=phenix_dir,
                        stderr=e,
                        stdout=o,
                    )

