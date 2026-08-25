import sys
from os.path import exists
from os import mkdir, environ
from subprocess import Popen
from abismal.callbacks import PhenixRunner


# phenix renamed the anomalous difference phase column from PHANOM to PANOM
# within the last couple of years, so which label the refined mtz carries
# depends on the local phenix version. Newer names come first; the first one
# present in the file wins.
ANOM_PHASE_KEYS = ('PANOM', 'PHANOM')

# Resolving the label has to happen after phenix.refine has written its mtz,
# which is inside the detached shell invocation rather than here -- hence a
# snippet rather than a function call.
_DETECT_PHASE_KEY = (
    "import glob, gemmi; "
    "labels = [c.label for c in "
    "gemmi.read_mtz_file(sorted(glob.glob('*[0-9].mtz'))[0]).columns]; "
    "print(next((k for k in {keys} if k in labels), {default}))"
)


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

        detect = _DETECT_PHASE_KEY.format(
            keys=repr(ANOM_PHASE_KEYS),
            default=repr(ANOM_PHASE_KEYS[0]),
        )
        command += f';PHASEKEY=$({sys.executable} -c "{detect}")'
        command += (
            f";rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p $PHASEKEY "
            f"-z {self.z_score_cutoff} -o peaks.csv"
        )

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

