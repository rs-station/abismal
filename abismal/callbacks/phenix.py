import tf_keras as tfk
import tensorflow as tf
from os.path import exists,dirname,abspath
from os import mkdir,environ
from subprocess import Popen,DEVNULL
from abismal.callbacks import MtzSaver
import gemmi

class PhenixRunner(tfk.callbacks.Callback):
    """ Run PHENIX periodically on the output. """
    def __init__(self, output_directory : str, eff_file : str, 
            epoch_stride: int=5, asu_id : int=0, output_prefix : str='phenix', *args, **kwargs):
        """
        Note that eff_file must use the $MTZFILE environment variable to point to the reflection file input.
	For example, the data manager block in the .eff file could look like the following

	```
	data_manager {
	  miller_array {
	    file = "$MTZFILE"
	    labels {
	      name = "F(+),SIGF(+),F(-),SIGF(-),merged"
	      array_type = amplitude
	    }
	    user_selected_labels = "F(+),SIGF(+),F(-),SIGF(-),merged"
	  }
	  miller_array {
	    file = "$ABISMALDIR/cxidb_81/reference_data/r-free-flags.mtz"
	    labels {
	      name = "R-free-flags"
	      array_type = integer
	    }
	    user_selected_labels = "R-free-flags"
	  }
	  fmodel {
	    xray_data {
	      outliers_rejection = False
	      french_wilson_scale = False
	    }
	  }
	  default_miller_array = "$MTZFILE"
	  model {
	    file = "$ABISMALDIR/cxidb_81/reference_data/2tli.pdb"
	  }
	  default_model = "$ABISMALDIR/cxidb_81/reference_data/2tli.pdb"
	}
	```

        """
        super().__init__(*args, **kwargs)
        self.output_prefix = output_prefix
        self.asu_id = asu_id
        self.eff_file = eff_file
        self.epoch_stride = epoch_stride
        self.output_directory = abspath(output_directory)
        self.processes = []

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_train_end(self, logs=None):
        for p in self.processes:
            p.wait()

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
        ]

        phenix_env = environ.copy()
        phenix_env['MTZFILE'] = mtz_file
        stderr = phenix_dir + '/stderr.txt'
        stdout = phenix_dir + '/stdout.txt'
        with open(stderr, 'w') as e, open(stdout, 'w') as o:
            p = Popen(
                command, 
                cwd=phenix_dir,
                stderr=e,
                stdout=o,
		env=phenix_env,
            )
            self.processes.append(p)

