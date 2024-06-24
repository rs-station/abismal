import reciprocalspaceship as rs
import tf_keras as tfk
import tensorflow as tf
from os.path import exists,dirname,abspath
from os import mkdir
from subprocess import Popen,DEVNULL
from abismal.callbacks import MtzSaver


class PhenixRunner(tfk.callbacks.Callback):
    """ Run PHENIX periodically on the output. """
    def __init__(self, output_directory, eff_file, 
            epoch_stride=5, asu_id=0, output_prefix='phenix', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_prefix = output_prefix
        self.asu_id = asu_id
        self.eff_file = eff_file
        self.epoch_stride = epoch_stride
        self.output_directory = abspath(output_directory)
        self.postprocessing

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_epoch_end(self, epoch, logs=None):
        if self.eff_file is not None and (epoch + 1) % self.epoch_stride == 0:
            self.run_phenix(epoch)

    def run_phenix(self, epoch):
        mtz_file = f"{self.output_directory}/asu_{self.asu_id}_epoch_{epoch+1}.mtz"

        phenix_dir = f"{self.output_directory}/{self.output_prefix}_asu_{self.asu_id}_epoch_{epoch+1}"
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

