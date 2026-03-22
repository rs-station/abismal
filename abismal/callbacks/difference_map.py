import tensorflow as tf
import tf_keras as tfk
from os.path import exists,dirname,abspath
from os import mkdir,listdir,environ
from subprocess import Popen,DEVNULL
from abismal.callbacks import PhenixRunner
from glob import glob


class DifferenceMap(PhenixRunner):
    """ Run PHENIX and anomalous peakfinding periodically on the output. """
    def __init__(self, output_directory, eff_file, 
                 epoch_stride=5, asu_id=0, output_prefix='phenix', z_score_cutoff=5., phase_key='PH2FOFCWT', compute_all_pairs=False, 
                 amp_key='F', sigma_key='SIGF', *args, **kwargs):
        """
        compute_all_pairs : bool
            optionally compute asu_1 - asu_0 and asu_0 - asu_1 maps. 
        """
        super().__init__(output_directory, eff_file, epoch_stride, asu_id, output_prefix, *args, **kwargs)
        self.z_score_cutoff=z_score_cutoff
        self.phase_key = phase_key
        self.amp_key = amp_key
        self.sigma_key = sigma_key
        self.compute_all_pairs = compute_all_pairs

    def run_phenix(self, epoch):
        epoch = epoch + 1 #1-indexed rather than zero
        mtz_file = f"{self.output_directory}/asu_{self.asu_id}_epoch_{epoch}.mtz"

        phenix_dir = f"{self.output_directory}/{self.output_prefix}_asu_{self.asu_id}_epoch_{epoch}"
        if not exists(phenix_dir):
            mkdir(phenix_dir)
        command = [
            'phenix.refine',
            self.eff_file,
            mtz_file,
        ]

        command = ' '.join(command)

        mtz_files = sorted(glob(f"{self.output_directory}/asu_*_epoch_{epoch}.mtz"), key=lambda x: int(x.split('_')[-3]))

        from itertools import combinations,permutations
        diffmap_dir = f"{self.output_directory}/diffmaps_epoch_{epoch}"
        if not exists(diffmap_dir):
            mkdir(diffmap_dir)
        if self.compute_all_pairs:
            it = permutations(mtz_files, 2)
        else:
            it = combinations(mtz_files, 2)
        for mtz_off,mtz_on in it:
            asu_on = mtz_on.split('_')[-3]
            asu_off = mtz_off.split('_')[-3]
            #The phase file will be an mtz in phenix_dir which does not end in `data.mtz`
            #we can glob for this using a negation for instance `ls *[!data].mtz` should 
            #only ever return a single file. 

            command += f';rs.diffmap -on {mtz_on} {self.amp_key} {self.sigma_key} -off {mtz_off} {self.amp_key} {self.sigma_key} -r {phenix_dir}/*[!data].mtz {self.phase_key} -o {diffmap_dir}/{asu_on}_minus_{asu_off}.mtz'

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

