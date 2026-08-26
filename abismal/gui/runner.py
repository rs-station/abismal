import html
import json
import threading
import glob
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path
import ipywidgets as widgets
from IPython.display import clear_output, display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abismal.gui.components.file_selector import _is_colab


# Building and saving a figure is not thread-safe, and two threads do it here: the
# tailer calls _poll directly while _schedule_poll's Timer calls it on its own
# thread. Matplotlib's unsafe state is global rather than per-figure -- the
# mathtext parser's pyparsing packrat cache above all -- so one lock covers every
# runner, not one each. The peak plot's "$\sigma$" label is what made this show
# up: without mathtext the race is silent and merely wasteful.
_FIGURE_LOCK = threading.Lock()


def _mtime(path):
    """Modification time, or None if the file went away mid-poll."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _is_loss_term(column):
    """The objective and the pieces it is made of.

    NLL and the KL terms are what `loss` is the sum of, so they belong on the
    same axes as it -- they are the only way to see which term a plateau or a
    divergence is coming from. KL_Sigma is matched by the KL prefix.
    """
    return 'loss' in column.lower() or column == 'NLL' or column.startswith('KL')


def _log_scale_is_safe(df, metrics):
    """Whether these metrics and their val_ partners can go on a log axis.

    A log axis silently drops non-positive points, and errors outright when none
    are left. NLL and the KL terms are positive in a healthy run, but nothing
    guarantees it for one that has gone wrong -- which is exactly when the plot
    is worth reading, so fall back to linear rather than hide the evidence.

    NaN is not a reason to: the `Epoch 0` row abismal writes from its
    pre-training callback is entirely NaN, and matplotlib draws that as a gap on
    either scale.
    """
    columns = [c for m in metrics for c in (m, f'val_{m}') if c in df.columns]
    values = df[columns].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return bool(finite.size and (finite > 0).all())


def _history_figure(df):
    """The training-history figure for `df`, or None if there is nothing to draw.

    Separate from _update_history so tests can assert on the axes -- scales,
    legends, line styles -- rather than on the PNG it ends up as. Rebuilding an
    equivalent figure in the tests instead would be free to drift from this one.

    Call it under _FIGURE_LOCK: matplotlib's unsafe state is global.
    """
    from matplotlib.figure import Figure

    panels = [
        (_base_metrics(df, _is_loss_term), 'Loss', 'Loss', True),
        (_base_metrics(df, lambda c: 'CC' in c), 'CC', None, False),
    ]
    panels = [p for p in panels if p[0]]
    if not panels:
        return None

    fig = Figure(figsize=(5 * len(panels), 4))
    axes = fig.subplots(1, len(panels))
    if len(panels) == 1:
        axes = [axes]

    for ax, (metrics, title, ylabel, log) in zip(axes, panels):
        _plot_metrics(ax, df, metrics)
        # The loss and its terms span orders of magnitude; CC runs through zero
        # and near it, where a log axis is meaningless.
        if log and _log_scale_is_safe(df, metrics):
            ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        _metric_legend(ax)

    fig.tight_layout()
    return fig


def _metric_legend(ax):
    """Legend the colours once, and explain solid vs dashed once.

    A `val_` line carries no colour of its own -- it is its metric's colour,
    dashed -- so listing both halves of every pair says the same thing twice and
    doubled the legend when NLL and the KL terms joined the panel. The lines keep
    their real labels; only what the legend is built from is filtered.
    """
    from matplotlib.lines import Line2D

    lines = ax.get_lines()
    handles = [l for l in lines if not l.get_label().startswith('val_')]
    labels = [l.get_label() for l in handles]
    if any(l.get_label().startswith('val_') for l in lines):
        handles += [
            Line2D([], [], color='0.35', ls='-'),
            Line2D([], [], color='0.35', ls='--'),
        ]
        labels += ['training', 'validation']
    ax.legend(handles, labels, fontsize='small', ncol=2, framealpha=0.85)


def _base_metrics(df, predicate):
    """Metric columns matching predicate, with their val_ partners folded in.

    Returns only the base names: `val_loss` is not a metric of its own, it is
    the validation half of `loss`, and the pair has to share a colour.
    """
    return [
        c for c in df.columns
        if predicate(c) and not c.startswith('val_') and c != 'Epoch'
    ]


def _plot_metrics(ax, df, metrics):
    """One Dark2 colour per metric; solid for training, dashed for validation."""
    import seaborn as sns

    # Ask for at least as many colours as metrics so the pairing survives a run
    # with more than the eight Dark2 provides -- it wraps, but consistently.
    palette = sns.color_palette('Dark2', max(len(metrics), 3))
    for color, base in zip(palette, metrics):
        ax.plot(df['Epoch'], df[base], color=color, ls='-', label=base)
        val = f'val_{base}'
        if val in df.columns:
            ax.plot(df['Epoch'], df[val], color=color, ls='--', label=val)
    ax.grid(which='both', axis='both', ls='-.')


class AbismalRunner:
    """
    Runs abismal as a detached subprocess with live output widgets.

    Output (stdout+stderr) is written to {out_dir}/console.log so it
    survives a kernel restart and can be reconnected via attach().
    """

    poll_interval = 10.0  # seconds

    def __init__(self, args, out_dir, has_phenix=False, total_epochs=None,
                 cwd=None):
        self.args = args
        self.out_dir = str(out_dir)
        self.has_phenix = has_phenix
        # What the child treats as "." -- the base directory the form resolves
        # its paths against, not the kernel's cwd, which is wherever the .ipynb
        # sits. None inherits, which is what a bare AbismalRunner should do.
        self.cwd = str(cwd) if cwd is not None else None
        self._pid = None
        self._process = None
        self._tailer_thread = None
        self._poll_timer = None
        self._last_pdb = None
        self._last_mtz = None
        self._peaks_signature = None

        self.console_log = os.path.join(self.out_dir, 'console.log')
        self.pid_file = os.path.join(self.out_dir, 'abismal.pid')

        self._log_text = ''
        self.log_widget = widgets.HTML(value=self._render_log_html())
        self.log_box = widgets.Box(
            [self.log_widget],
            layout=widgets.Layout(
                height='300px',
                overflow_y='auto',
                border='1px solid #ccc',
            ),
        )
        self.log_box.add_class('abismal-log-scroll')
        self._log_js_widget = widgets.Output(
            layout=widgets.Layout(height='0px', overflow='hidden'),
        )
        self._init_log_autoscroll_js()
        self.progress_widget = widgets.IntProgress(
            min=0,
            max=total_epochs or 1,
            value=0,
            bar_style='info',
            layout=widgets.Layout(flex='1'),
        )
        self.progress_label = widgets.Label(
            'Waiting...', layout=widgets.Layout(min_width='120px')
        )
        self.stop_button = widgets.Button(
            description='Stop',
            button_style='danger',
            tooltip='Terminate training',
        )
        self.stop_button.on_click(lambda _: self.stop())
        self.history_widget = widgets.Output()
        self.peaks_widget = widgets.Output()
        # Hidden until there is something to show: only anomalous refinement runs
        # write peaks.csv, and we cannot know that until one appears on disk.
        self.peaks_label = widgets.HTML(
            value="<b>Anomalous Peak Heights</b>",
            layout=widgets.Layout(display='none'),
        )
        if has_phenix:
            self._viewer_id = str(uuid.uuid4())
            self._viewer_initialized = False
            self.viewer_widget = widgets.Output()
            self.viewer_label = widgets.HTML(value='')
            self._js_widget = widgets.Output(
                layout=widgets.Layout(height='0px', overflow='hidden')
            )
        else:
            self._viewer_id = None
            self._viewer_initialized = False
            self.viewer_widget = None
            self.viewer_label = None
            self._js_widget = None

        # Colab: kernel→frontend sync of widget traits from background threads
        # is broken. We work around it by having a JS interval call back into
        # the kernel, and using each invocation (which runs in event-loop
        # context) to force-resync the widget state to the frontend.
        self._monitoring_active = True
        self._colab_poll_widget = widgets.Output(
            layout=widgets.Layout(height='0px', overflow='hidden'),
        )
        if _is_colab():
            self._setup_colab_polling()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Launch a fresh abismal subprocess."""
        os.makedirs(self.out_dir, exist_ok=True)
        cmd = ['abismal', *self.args, '--keras-verbosity', '2']
        # Force unbuffered stdout/stderr in the child so console.log fills in
        # real time — without this the tail thread sees nothing until the
        # child's block buffer flushes (often only on exit).
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        fout = open(self.console_log, 'w')
        self._process = subprocess.Popen(
            cmd,
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            cwd=self.cwd,
        )
        fout.close()
        self._pid = self._process.pid
        with open(self.pid_file, 'w') as f:
            f.write(str(self._pid))
        self._begin_monitoring()

    @classmethod
    def attach(cls, out_dir, has_phenix=False):
        """Return a runner attached to a live orphaned process, or None."""
        pid_path = os.path.join(str(out_dir), 'abismal.pid')
        if not os.path.exists(pid_path):
            return None
        try:
            pid = int(Path(pid_path).read_text().strip())
        except (ValueError, OSError):
            return None
        if not cls._pid_is_abismal(pid):
            return None
        runner = cls(args=None, out_dir=out_dir, has_phenix=has_phenix)
        runner._pid = pid
        return runner

    def resume(self):
        """Start monitoring an attached (previously orphaned) process."""
        self._begin_monitoring()

    def stop(self):
        """Terminate the subprocess; runs the wait loop in a daemon thread."""
        if self._pid is None:
            return

        def _do_stop():
            try:
                os.kill(self._pid, signal.SIGTERM)
                for _ in range(20):
                    if not self._pid_is_abismal(self._pid):
                        return
                    time.sleep(0.5)
                os.kill(self._pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        threading.Thread(target=_do_stop, daemon=True).start()

    def shutdown(self, timeout=5.0):
        """Stop monitoring and release this runner's threads. Idempotent.

        Nothing else cancels the poll Timer. `_schedule_poll` re-arms itself on every
        tick and only stops re-arming once `is_running` goes false, so an already-armed
        timer always survives. Clearing `_monitoring_active` is what stops the two
        things that outlive the child: `_post_training_phenix_watcher`, which otherwise
        keeps reading out_dir for two minutes after the job ends, and on Colab the
        browser interval that syncs this runner's widgets. Without this, re-clicking Run
        accumulates runners that go on polling a directory the next run is overwriting.

        This does not terminate the subprocess: `stop()` does that, and an attached job
        is deliberately allowed to outlive the kernel.
        """
        self._monitoring_active = False
        timer = self._poll_timer
        if timer is not None:
            timer.cancel()
            self._poll_timer = None
        thread = self._tailer_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    @property
    def is_running(self):
        # For a child we launched, Popen is authoritative. The /proc check below has a
        # window it cannot see through: between fork and exec, /proc/<pid>/cmdline is
        # momentarily zero-length, so `b'abismal' in cmdline` is False for a process
        # that is very much alive. The tailer reads that as "the run is over", drains a
        # console.log nothing has been written to yet, and stops -- leaving an empty log
        # for a job that ran. Measured at roughly 1 run in 30 for a job that finishes in
        # ~50 ms, which is to say precisely the fast-failing runs whose log matters most.
        if self._process is not None:
            return self._process.poll() is None
        # An attached job is not our child, so /proc is all there is. The window is not
        # reachable there anyway: attach() only ever sees a process that has long since
        # exec'd.
        return self._pid is not None and self._pid_is_abismal(self._pid)

    def to_widget(self):
        progress_row = widgets.HBox([
            self.progress_widget,
            self.progress_label,
            self.stop_button,
        ])
        sections = [
            widgets.HTML("<b>Progress</b>"),
            progress_row,
            widgets.HTML("<b>Training History</b>"),
            self.history_widget,
            self.peaks_label,
            self.peaks_widget,
        ]
        if self.has_phenix:
            sections += [
                widgets.HTML("<b>Refinement Results</b>"),
                self.viewer_label,
                self.viewer_widget,
                self._js_widget,
            ]
        sections += [
            widgets.HTML("<b>Log Output</b>"),
            self.log_box,
            self._log_js_widget,
            self._colab_poll_widget,
        ]
        return widgets.VBox(sections)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _pid_is_abismal(pid):
        """Check whether pid is a live abismal process. Linux /proc only."""
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
            return b'abismal' in cmdline
        except OSError:
            return False

    def _render_log_html(self):
        return (
            '<pre style="margin:0;font-family:monospace;font-size:12px;'
            'white-space:pre-wrap;line-height:1.3;">'
            + html.escape(self._log_text)
            + '</pre>'
        )

    def _run_on_main_thread(self, fn):
        """Schedule a widget mutation on the kernel's main event loop. On
        Colab, comm messages from background threads don't reliably sync to
        the frontend, so widget trait changes have to be marshaled onto the
        io_loop. On Jupyter this is a no-op-equivalent (still runs fn)."""
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                kernel = getattr(ip, 'kernel', None)
                if kernel is not None and hasattr(kernel, 'io_loop'):
                    kernel.io_loop.add_callback(fn)
                    return
        except Exception:
            pass
        fn()

    def _append_log(self, line):
        self._log_text += line
        html_val = self._render_log_html()
        self._run_on_main_thread(
            lambda: setattr(self.log_widget, 'value', html_val)
        )

    def _init_log_autoscroll_js(self):
        js = """
        (function() {
            function attach(box) {
                if (box.__abismal_autoscroll) return;
                box.__abismal_autoscroll = true;
                var stickBottom = true;
                var ignoreNext = false;
                box.addEventListener('scroll', function() {
                    if (ignoreNext) { ignoreNext = false; return; }
                    stickBottom = (box.scrollTop + box.clientHeight)
                                  >= (box.scrollHeight - 5);
                });
                var observer = new MutationObserver(function() {
                    if (stickBottom) {
                        ignoreNext = true;
                        box.scrollTop = box.scrollHeight;
                    }
                });
                observer.observe(box, {
                    childList: true, subtree: true, characterData: true,
                });
                ignoreNext = true;
                box.scrollTop = box.scrollHeight;
            }
            var attempts = 0;
            var iv = setInterval(function() {
                document.querySelectorAll('.abismal-log-scroll').forEach(attach);
                if (++attempts > 50) clearInterval(iv);
            }, 100);
        })();
        """
        self._log_js_widget.outputs = ({
            'output_type': 'display_data',
            'data': {'application/javascript': js},
            'metadata': {},
        },)

    def _begin_monitoring(self):
        self._monitoring_active = True
        self._tailer_thread = threading.Thread(target=self._tail, daemon=True)
        self._tailer_thread.start()
        self._schedule_poll()

    def _setup_colab_polling(self):
        """Register a Colab kernel callback and inject JS to drive periodic
        state sync. Each callback invocation runs in event-loop context, so
        the widget.send_state calls actually push to the frontend."""
        try:
            from google.colab import output as _colab_output
        except ImportError:
            return

        poll_id = f'abismal_runner_poll_{id(self)}'

        def _poll_callback():
            try:
                widgets_and_traits = [
                    (self.log_widget, ['value']),
                    (self.progress_widget, ['value', 'max', 'bar_style']),
                    (self.progress_label, ['value']),
                    (self.stop_button, ['disabled']),
                    (self.history_widget, ['outputs']),
                    (self.peaks_label.layout, ['display']),
                    (self.peaks_widget, ['outputs']),
                ]
                if self.has_phenix:
                    widgets_and_traits += [
                        (self.viewer_label, ['value']),
                        (self.viewer_widget, ['outputs']),
                        (self._js_widget, ['outputs']),
                    ]
                for w, traits in widgets_and_traits:
                    try:
                        w.send_state(traits)
                    except Exception:
                        pass
            except Exception:
                pass
            return self._monitoring_active

        _colab_output.register_callback(poll_id, _poll_callback)

        from string import Template
        js = Template("""
        (function() {
          var iv = setInterval(async function() {
            try {
              var r = await google.colab.kernel.invokeFunction(
                  '$POLL_ID', [], {});
              if (r && r.data && r.data['application/json'] === false) {
                try {
                  await google.colab.kernel.invokeFunction(
                      '$POLL_ID', [], {});
                } catch (e) {}
                clearInterval(iv);
              }
            } catch (e) { clearInterval(iv); }
          }, 1000);
        })();
        """).substitute(POLL_ID=poll_id)
        self._colab_poll_widget.outputs = ({
            'output_type': 'display_data',
            'data': {'application/javascript': js},
            'metadata': {},
        },)

    def _update_progress(self, cur, total):
        def apply():
            self.progress_widget.max = total
            self.progress_widget.value = cur - 1
            self.progress_label.value = f'Epoch {cur} / {total}'
        self._run_on_main_thread(apply)

    def _tail(self):
        """Stream console.log into log_widget and drive the progress bar."""
        epoch_re = re.compile(r'Epoch (\d+)/(\d+)')
        with open(self.console_log, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    self._append_log(line)
                    m = epoch_re.match(line)
                    if m:
                        self._update_progress(int(m.group(1)), int(m.group(2)))
                elif self.is_running:
                    time.sleep(0.5)
                else:
                    # Process done — drain any lines written just before exit.
                    for line in f:
                        self._append_log(line)
                        m = epoch_re.match(line)
                        if m:
                            self._update_progress(
                                int(m.group(1)), int(m.group(2))
                            )
                    break

        self._run_on_main_thread(
            lambda: setattr(self.stop_button, 'disabled', True)
        )

        if self._process is not None:
            rc = self._process.wait()
            if rc == 0:
                def finish_ok():
                    self.progress_widget.value = self.progress_widget.max
                    self.progress_widget.bar_style = 'success'
                    self.progress_label.value = 'Finished'
                self._run_on_main_thread(finish_ok)
            else:
                def finish_fail(rc=rc):
                    self.progress_widget.bar_style = 'danger'
                    self.progress_label.value = f'Failed (exit {rc})'
                self._run_on_main_thread(finish_fail)
        else:
            def finish_unknown():
                self.progress_widget.bar_style = ''
                self.progress_label.value = 'Process exited (check abismal.log)'
            self._run_on_main_thread(finish_unknown)

        try:
            os.remove(self.pid_file)
        except OSError:
            pass

        self._poll()

        if self.has_phenix:
            threading.Thread(
                target=self._post_training_phenix_watcher, daemon=True
            ).start()
        else:
            # No phenix watcher → monitoring is done; tell the Colab poll
            # loop it can stop after one final sync.
            self._monitoring_active = False

    def _schedule_poll(self):
        if not self.is_running:
            return
        self._poll()
        self._poll_timer = threading.Timer(self.poll_interval, self._schedule_poll)
        self._poll_timer.daemon = True
        self._poll_timer.start()

    def _poll(self):
        self._update_history()
        if self.has_phenix:
            self._update_viewer()
            self._update_peaks()

    def _update_history(self):
        history_file = Path(self.out_dir) / "history.csv"
        if not history_file.exists():
            return
        try:
            df = pd.read_csv(history_file)
        except Exception:
            return
        if df.empty:
            return

        with _FIGURE_LOCK:
            fig = _history_figure(df)
            if fig is None:
                return
            self._show_figure(self.history_widget, fig)

    def _update_peaks(self):
        """Plot anomalous peak height against epoch, coloured by atom type and
        styled by residue.

        Only anomalous runs with refinement produce peaks.csv -- phenix via
        AnomalousPeakFinder, torchref via its worker -- so the files being there
        at all is the signal that this plot applies. Nothing is drawn otherwise,
        and the section stays hidden.
        """
        files = self._peaks_files()
        if not files:
            return
        # Drawing this is not cheap and the poll runs on a timer, so only redraw
        # when a peaks.csv has actually appeared or changed. Without the guard a
        # finished run keeps re-rendering the same figure for as long as the
        # widget is alive.
        signature = tuple((f, _mtime(f)) for f in files)
        if signature == self._peaks_signature:
            return

        peak_data = self._read_peaks(files)
        if peak_data is None:
            return

        # Drop peaks seen in only a handful of epochs: they are noise excursions
        # rather than sites. report.py uses a flat 10-epoch floor, which no run
        # clears while it is still going, so scale it to what has been seen.
        n_epochs = peak_data['Epoch'].nunique()
        min_points = max(1, round(0.5 * n_epochs))
        counts = peak_data.groupby('Residue')['Epoch'].transform('size')
        peak_data = peak_data[counts >= min_points]
        if peak_data.empty:
            return

        import seaborn as sns
        from matplotlib.figure import Figure

        with _FIGURE_LOCK:
            fig = Figure(figsize=(7, 4))
            ax = fig.subplots()
            sns.lineplot(
                peak_data, x='Epoch', y='peakz', hue='AtomType', style='Residue',
                palette='Dark2', ax=ax,
            )
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            ax.grid(which='both', axis='both', ls='-.')
            ax.set_ylabel(r"Anomalous Peak Height ($\sigma$)")
            fig.tight_layout()

            self._show_figure(self.peaks_widget, fig)
        self._peaks_signature = signature
        self._run_on_main_thread(
            lambda: setattr(self.peaks_label.layout, 'display', '')
        )

    def _peaks_files(self):
        """Every peaks.csv under out_dir, from either refinement backend."""
        files = []
        for prefix in ("eff", "torchref"):
            pattern = str(Path(self.out_dir) / f"{prefix}_*_asu_*_epoch_*" / "peaks.csv")
            files.extend(glob.glob(pattern))
        return sorted(files)

    def _read_peaks(self, files=None):
        """Every peaks.csv under out_dir, tagged with its epoch. None if empty."""
        if files is None:
            files = self._peaks_files()
        frames = []
        for path in files:
            try:
                epoch = int(Path(path).parent.name.split('_epoch_')[-1])
            except ValueError:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or not {'chain', 'seqid', 'residue', 'peakz'} <= set(df.columns):
                continue
            df = df[['chain', 'seqid', 'residue', 'peakz']].copy()
            df['Epoch'] = epoch
            frames.append(df)
        if not frames:
            return None
        data = pd.concat(frames, ignore_index=True)
        data['Residue'] = (
            data['residue'] + '-' + data['seqid'].astype(str) + ':' + data['chain']
        )
        # peaks.csv also carries `name`, the atom name -- but in every run seen
        # so far (phenix's AnomalousPeakFinder and the torchref worker alike)
        # it comes out empty, so it implies no element to colour by. The residue
        # type (CYS, MET, ...) is what actually distinguishes anomalous
        # scatterers in practice, so that is the atom-type key instead, kept
        # apart from `Residue` (the specific residue instance) so the plot can
        # colour by one and style by the other.
        data['AtomType'] = data['residue']
        return data

    def _show_figure(self, widget, fig):
        """Render fig into an Output widget's outputs, from any thread."""
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        new_outputs = ({
            'output_type': 'display_data',
            'data': {'image/png': png_b64},
            'metadata': {},
        },)
        self._run_on_main_thread(
            lambda: setattr(widget, 'outputs', new_outputs)
        )


    def _find_latest_phenix_results(self, asu_id=0):
        # Match both phenix (eff_*) and torchref (torchref_*) result dirs.
        dirs = []
        for prefix in ("eff", "torchref"):
            pattern = str(
                Path(self.out_dir) / f"{prefix}_*_asu_{asu_id}_epoch_*"
            )
            dirs.extend(glob.glob(pattern))
        if not dirs:
            return None, None

        def epoch_key(d):
            try:
                return int(d.split('_epoch_')[-1])
            except (ValueError, IndexError):
                return -1

        dirs.sort(key=epoch_key)
        for d in reversed(dirs):
            pdb_files = glob.glob(str(Path(d) / "*.pdb"))
            mtz_files = [f for f in glob.glob(str(Path(d) / "*.mtz"))
                         if not f.endswith('data.mtz')]
            if pdb_files and mtz_files:
                return pdb_files[0], mtz_files[0]
        return None, None

    def _update_viewer(self):
        pdb_file, mtz_file = self._find_latest_phenix_results()
        if pdb_file is None:
            return
        if pdb_file == self._last_pdb and mtz_file == self._last_mtz:
            return
        self._render_epoch(pdb_file, mtz_file)

    def _render_epoch(self, pdb_file, mtz_file):
        try:
            from abismal.gui.components.gemmimol import GemmiMolViewer
            viewer = GemmiMolViewer(
                pdb_file=pdb_file, mtz_file=mtz_file,
                viewer_id=self._viewer_id,
            )
            # Reads and encodes both files, so it also settles whether they are
            # still there and complete.
            payload = viewer.reload_payload
        except Exception:
            # Still being written by phenix, or deleted under us — retry on the
            # next poll.
            return

        # `Overwrite and Run` rmtree's result directories while a poll may be in
        # flight. Encoding above reads both files, so one that has vanished raises
        # there and is retried; this catches a directory emptied between the two
        # reads. Getting it wrong used to leave the viewer on "Loading..." with
        # nothing to retry, since _update_viewer skips whatever _last_pdb names.
        if not (os.path.exists(pdb_file) and os.path.exists(mtz_file)):
            return

        # Advance cache only on success so a partial write doesn't block retries.
        self._last_pdb = pdb_file
        self._last_mtz = mtz_file

        epoch = Path(pdb_file).parent.name.split('_epoch_')[-1]
        self.viewer_label.value = (
            f'<b>Epoch {epoch}</b> &nbsp;|&nbsp; '
            f'<code>{html.escape(Path(pdb_file).name)}</code> &nbsp;+&nbsp; '
            f'<code>{html.escape(Path(mtz_file).name)}</code>'
        )

        if not self._viewer_initialized:
            # First render: create the iframe; subsequent renders will postMessage it.
            self.viewer_widget.outputs = ({
                'output_type': 'display_data',
                'data': {'text/html': viewer.html},
                'metadata': {'text/html': {'isolated': True}},
            },)
            self._viewer_initialized = True
        else:
            # Find the iframe by its ABISMAL_VIEWER_ID and ask it to reload in-place,
            # preserving camera orientation.
            js = (
                f'/*{time.time()}*/(function(){{'
                f'var t=Array.from(document.querySelectorAll("iframe")).find('
                f'function(f){{try{{return f.contentWindow.ABISMAL_VIEWER_ID==='
                f'"{self._viewer_id}";}}catch(e){{return false;}}}});'
                f'if(t)t.contentWindow.postMessage({json.dumps(payload)},"*");'
                f'}})();'
            )
            self._js_widget.outputs = ({
                'output_type': 'display_data',
                'data': {'application/javascript': js},
                'metadata': {},
            },)

    def _post_training_phenix_watcher(self, max_unchanged=12):
        """Keep polling for Phenix results after training exits (Phenix may still be running).

        `_monitoring_active` is the loop's other exit: this thread outlives the child by
        up to `max_unchanged * poll_interval` -- two minutes by default -- and shutdown()
        has to be able to stop it, or a runner the form has replaced goes on reading an
        out_dir the next run is about to overwrite.
        """
        unchanged = 0
        while self._monitoring_active and unchanged < max_unchanged:
            time.sleep(self.poll_interval)
            prev = self._last_pdb
            self._update_viewer()
            if self._last_pdb == prev:
                unchanged += 1
            else:
                unchanged = 0
        # Nothing polls after this, so the Colab sync loop has no reason to keep
        # calling back into the kernel. _tail already does this on the branch with no
        # refinement; this is the same signal for the branch that has one.
        self._monitoring_active = False
