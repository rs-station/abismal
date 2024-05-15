import reciprocalspaceship as rs
import tensorflow as tf
import tf_keras as tfk
from os.path import exists,dirname,abspath
from os import mkdir,listdir
from subprocess import Popen,DEVNULL
from abismal.callbacks import MtzSaver

#TODO: refactor all the other mtz saving callbacks as special cases of this one
class AnomalousPeakFinder(tfk.callbacks.Callback):
    """ A hybrid callback that saves mtzs, runs phenix, and runs peakfinding """
    def __init__(self, output_directory, eff_file, epoch_stride=10, z_score_cutoff=5., **kwargs):
        super().__init__(**kwargs)
        self.output_directory = abspath(output_directory)
        self.z_score_cutoff = z_score_cutoff
        self.eff_file = eff_file
        self.epoch_stride = epoch_stride

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_stride == 0:
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
                command = ' '.join(command)
                command += f";rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PHANOM -z {self.z_score_cutoff} -o peaks.csv"

                stderr = phenix_dir + '/stderr.txt'
                stdout = phenix_dir + '/stdout.txt'
                with open(stderr, 'w') as e, open(stdout, 'w') as o:
                    p = Popen(
                        command, 
                        shell=True,
                        cwd=phenix_dir,
                        stderr=e,
                        stdout=o,
                    )

