import sys
import tf_keras as tfk
from os.path import exists, abspath, dirname, join
from os import mkdir, environ
from subprocess import Popen


class TorchRefRunner(tfk.callbacks.Callback):
    """Run torchref refinement periodically on the merged output.

    A PyTorch-based alternative to :class:`PhenixRunner`. Because torchref runs
    on CPU/GPU through PyTorch (no external Phenix install), it works on Colab.

    Each invocation launches ``_torchref_worker.py`` as a detached subprocess.
    The worker is started *by file path* so it never imports the abismal package
    (and therefore never initializes a second TensorFlow/CUDA context); it is a
    pure PyTorch process. Refinement runs on CPU by default to avoid contending
    with the training process for GPU memory.

    When the merged data are anomalous the worker also builds an anomalous
    difference map and runs peak finding on it, writing ``peaks.csv`` alongside
    the refined model. That happens automatically -- anomalous data is detected
    from the MTZ columns, so there is no separate flag to enable it. This is the
    torchref counterpart to :class:`AnomalousPeakFinder`.
    """

    def __init__(self, output_directory: str, pdb_file: str,
                 epoch_stride: int = 1, asu_id: int = 0,
                 output_prefix: str = 'torchref', device: str = 'cpu',
                 macro_cycles: int = 5, z_score_cutoff: float = 5.,
                 r_free_mtz: str = None, r_free_value: int = None,
                 wavelength: float = None, adp_mode: str = 'auto',
                 adp_aniso_sigma: str = 'auto', rigid_body: bool = True,
                 rigid_body_iter: int = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_prefix = output_prefix
        self.asu_id = asu_id
        self.pdb_file = abspath(pdb_file) if pdb_file is not None else None
        self.epoch_stride = epoch_stride
        self.device = device
        self.macro_cycles = macro_cycles
        self.z_score_cutoff = z_score_cutoff
        self.r_free_mtz = abspath(r_free_mtz) if r_free_mtz is not None else None
        self.r_free_value = r_free_value
        self.wavelength = wavelength
        self.adp_mode = adp_mode
        self.adp_aniso_sigma = adp_aniso_sigma
        self.rigid_body = rigid_body
        self.rigid_body_iter = rigid_body_iter
        self.output_directory = abspath(output_directory)
        self.processes = []

        if not exists(self.output_directory):
            mkdir(output_directory)

    def on_train_end(self, logs=None):
        for p in self.processes:
            p.wait()

    def on_epoch_end(self, epoch, logs=None):
        if self.pdb_file is not None and (epoch + 1) % self.epoch_stride == 0:
            self.run_torchref(epoch)

    def run_torchref(self, epoch):
        mtz_file = f"{self.output_directory}/asu_{self.asu_id}_epoch_{epoch+1}.mtz"

        result_dir = (
            f"{self.output_directory}/{self.output_prefix}"
            f"_asu_{self.asu_id}_epoch_{epoch+1}"
        )
        if not exists(result_dir):
            mkdir(result_dir)

        worker = join(dirname(abspath(__file__)), "_torchref_worker.py")
        # On Colab, abismal and torchref share one environment, so the training
        # interpreter (sys.executable) can run the worker directly. When they
        # live in separate environments (common for local Jupyter), point
        # ABISMAL_TORCHREF_PYTHON at an interpreter that has torchref installed.
        python_exe = environ.get("ABISMAL_TORCHREF_PYTHON", sys.executable)
        command = [
            python_exe, worker,
            "--mtz", mtz_file,
            "--pdb", self.pdb_file,
            "--out-dir", result_dir,
            "--device", self.device,
            "--macro-cycles", str(self.macro_cycles),
            "--z-score-cutoff", str(self.z_score_cutoff),
        ]
        if self.r_free_mtz is not None:
            command += ["--r-free-mtz", self.r_free_mtz]
            if self.r_free_value is not None:
                command += ["--r-free-value", str(self.r_free_value)]
        if self.wavelength is not None:
            command += ["--wavelength", str(self.wavelength)]
        command += ["--adp-mode", str(self.adp_mode),
                    "--adp-aniso-sigma", str(self.adp_aniso_sigma),
                    "--rigid-body-iter", str(self.rigid_body_iter)]
        if not self.rigid_body:
            command += ["--no-rigid-body"]

        stderr = join(result_dir, "stderr.txt")
        stdout = join(result_dir, "stdout.txt")
        with open(stderr, 'w') as e, open(stdout, 'w') as o:
            p = Popen(command, cwd=result_dir, stderr=e, stdout=o)
            self.processes.append(p)
