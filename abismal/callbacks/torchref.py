import sys
import tf_keras as tfk
from os.path import exists, abspath, dirname, join
from os import mkdir, environ
from subprocess import Popen
from warnings import warn


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
                 rigid_body_iter: int = 30, allow_overlap: bool = False,
                 *args, **kwargs):
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
        self.allow_overlap = allow_overlap
        self.output_directory = abspath(output_directory)
        self.processes = []
        # Epoch whose refinement the overlap guard skipped, if that skip is
        # still the most recent one. on_train_end refines it. See there.
        self._skipped_final_epoch = None

        if not exists(self.output_directory):
            mkdir(output_directory)

    # Concurrent workers tolerated under allow_overlap before saying something.
    # Two is normal on the benchmarks; this is well clear of it.
    OVERLAP_WARN_AT = 4

    def _reap(self, block=False):
        """Collect finished workers, reporting any that failed.

        Called every epoch. Without it `self.processes` grows without bound and
        each finished worker stays a zombie for the length of training -- one
        per epoch at the default `--torchref-frequency 1`.

        A worker failure is otherwise invisible: it writes to `stderr.txt`
        inside its own result directory and training carries on reporting
        success. Surface the returncode instead, once, with the path to look in.
        """
        still_running = []
        for process, result_dir in self.processes:
            if block:
                process.wait()
            elif process.poll() is None:
                still_running.append((process, result_dir))
                continue
            if process.returncode:
                warn(
                    f"torchref worker for {result_dir} exited with "
                    f"{process.returncode}; see {join(result_dir, 'stderr.txt')}",
                    RuntimeWarning,
                )
        self.processes = still_running

    def on_train_end(self, logs=None):
        """Wait out the running workers, then refine the final epoch if it was skipped.

        The final epoch is the headline result -- it is the model the run is
        judged on -- so it is the one epoch that must not be lost to the overlap
        guard. Whether it is lost is otherwise a coin flip: measured on the
        banked benchmarks, a worker takes 1.33x an epoch on hewl and 0.97x on
        cxidb_61, so those skip roughly every other epoch and whichever way the
        parity falls decides whether the last one ran.

        Blocking here is free -- training is over.
        """
        self._reap(block=True)
        if self._skipped_final_epoch is not None:
            epoch = self._skipped_final_epoch
            self._skipped_final_epoch = None
            self.run_torchref(epoch)
            self._reap(block=True)

    def on_epoch_end(self, epoch, logs=None):
        self._reap()
        if self.pdb_file is None or (epoch + 1) % self.epoch_stride:
            return
        if self.processes and not self.allow_overlap:
            # Remember the most recent skip. If training ends here, on_train_end
            # picks it up rather than leaving the run without its final result.
            self._skipped_final_epoch = epoch
            # Refinement is CPU-bound and can outlast an epoch. Letting runs
            # pile up would put N of them on the box at once, each fighting the
            # others and the trainer for cores, so skip instead. The next
            # multiple of epoch_stride will pick up better-converged data
            # anyway. Set allow_overlap to refine every epoch regardless.
            warn(
                f"skipping torchref at epoch {epoch + 1}: "
                f"{len(self.processes)} earlier run(s) still going",
                RuntimeWarning,
            )
            return
        if len(self.processes) >= self.OVERLAP_WARN_AT:
            # Steady-state concurrency is roughly worker_time / epoch_time, so
            # it is self-limiting on the benchmarks (~2 at worst). Well past
            # that means refinement is far slower than an epoch and the workers
            # are contending with each other; say so rather than block, since
            # the caller asked for every epoch.
            warn(
                f"{len(self.processes)} torchref runs already going at epoch "
                f"{epoch + 1}; refinement is much slower than an epoch, so "
                "these are competing for cores. Consider --torchref-frequency.",
                RuntimeWarning,
            )
        self._skipped_final_epoch = None
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
            self.processes.append((p, result_dir))
