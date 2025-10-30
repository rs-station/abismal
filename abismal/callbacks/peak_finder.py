import tensorflow as tf
import tf_keras as tfk
from os.path import exists,dirname,abspath
from os import mkdir,listdir,environ
from subprocess import Popen,DEVNULL
from abismal.callbacks import PhenixRunner


class AnomalousPeakFinder(PhenixRunner):
    """ Run PHENIX and anomalous peakfinding periodically on the output. """
    def __init__(self, output_directory, eff_file, 
            epoch_stride=5, asu_id=0, output_prefix='phenix', z_score_cutoff=5., *args, **kwargs):
        super().__init__(output_directory, eff_file, epoch_stride, asu_id, output_prefix, *args, **kwargs)
        self.z_score_cutoff=z_score_cutoff

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

        command = ' '.join(command)
        command += f";rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z {self.z_score_cutoff} -o peaks.csv"

        phenix_env = environ.copy()
        phenix_env['MTZFILE'] = mtz_file
        stderr = phenix_dir + '/stderr.txt'
        stdout = phenix_dir + '/stdout.txt'
        with open(stderr, 'w') as e, open(stdout, 'w') as o:
            p = Popen(
                command, 
                shell=True,
                cwd=phenix_dir,
                stderr=e,
                stdout=o,
                env=phenix_env,
            )
            self.processes.append(p)

