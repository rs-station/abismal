import html
import json
import math
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
import pandas as pd
import matplotlib.pyplot as plt

from abismal.gui._log_filter import is_noise
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
        # and near it, where a log axis is meaningless. NLL goes negative often
        # enough that those points are simply not drawn -- matplotlib clips them
        # and carries on, and seeing the rest on a log axis is worth more than
        # seeing all of it on a linear one.
        if log:
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


def _lone_point_marker(values):
    """A marker for a series with a single point, which a line alone cannot show.

    After the first epoch every series is one point; lines stay unmarked once
    there are two.
    """
    return 'o' if values.notna().sum() == 1 else None


def _plot_metrics(ax, df, metrics):
    """One Dark2 colour per metric; solid for training, dashed for validation."""
    import seaborn as sns

    # Ask for at least as many colours as metrics so the pairing survives a run
    # with more than the eight Dark2 provides -- it wraps, but consistently.
    palette = sns.color_palette('Dark2', max(len(metrics), 3))
    for color, base in zip(palette, metrics):
        ax.plot(df['Epoch'], df[base], color=color, ls='-', label=base,
                marker=_lone_point_marker(df[base]))
        val = f'val_{base}'
        if val in df.columns:
            ax.plot(df['Epoch'], df[val], color=color, ls='--', label=val,
                    marker=_lone_point_marker(df[val]))
    ax.grid(which='both', axis='both', ls='-.')


_STATUS_RE = re.compile(r'^\[status\] (.+?)\s*$')
_REFINEMENT_RE = re.compile(r'^\[refinement\] (started|finished|failed|skipped) (\S+)')
_CYCLE_RE = re.compile(r'^Cycle\s+(\d+)/(\d+):.*Rfree=([0-9.]+)')


def _refinement_stage(result_dir):
    """What a running torchref worker is doing, read from the end of its log.

    The worker's stdout marks each stage as it starts; the last marker seen is the
    stage it is in.
    """
    try:
        lines = (Path(result_dir) / 'stdout.txt').read_text().splitlines()
    except OSError:
        return 'starting'
    stage = 'setting up'
    for line in lines:
        cycle = _CYCLE_RE.match(line)
        if cycle:
            stage = f'macrocycle {cycle.group(1)}/{cycle.group(2)} (R<sub>free</sub> {cycle.group(3)})'
        elif 'fitting SIMU' in line or line.lstrip().startswith('sigma='):
            stage = 'fitting the ADP restraint weight'
        elif 'rigid-body' in line or line.startswith('Rigid body'):
            stage = 'rigid-body refinement'
        elif 'anomalous peaks' in line or line.startswith('peak_local_max'):
            stage = 'finding anomalous peaks'
    return stage


def _epoch_of(name):
    try:
        return int(name.rsplit('_epoch_', 1)[-1])
    except ValueError:
        return None


_R_WORKER = re.compile(r'^R(work|free)=([0-9.]+)\s*$', re.M)
_R_PHENIX = re.compile(r'Final R-work = ([0-9.]+), R-free = ([0-9.]+)')


def _refinement_r_factors(result_dir):
    """(Rwork, Rfree) of one refinement result directory, or None.

    The torchref worker prints them as its last `Rwork=` and `Rfree=` lines.
    Not its PDB header: torchref before 0.7.0 copied REMARK 3 from the input
    model, so the header there reports the starting model's R values. phenix
    logs `Final R-work = ..., R-free = ...`, and writes the same numbers into
    its model's REMARK 3, read here when the log is gone.
    """
    d = Path(result_dir)
    if d.name.startswith('torchref'):
        try:
            found = dict(_R_WORKER.findall((d / 'stdout.txt').read_text()))
        except OSError:
            return None
        if 'work' in found and 'free' in found:
            return float(found['work']), float(found['free'])
        return None
    for log in sorted(d.glob('*.log')):
        try:
            matches = _R_PHENIX.findall(log.read_text())
        except OSError:
            continue
        if matches:
            return tuple(float(v) for v in matches[-1])
    for pdb in sorted(d.glob('*.pdb')):
        try:
            import gemmi
            refinement = gemmi.read_pdb(str(pdb)).meta.refinement
        except Exception:
            continue
        # gemmi reports a value missing from the header as NaN, which is truthy.
        if refinement and all(math.isfinite(v) and v > 0 for v in (refinement[0].r_work, refinement[0].r_free)):
            return refinement[0].r_work, refinement[0].r_free
    return None


def _draw_r_factors(ax, data):
    """Rwork (solid) and Rfree (dashed) against epoch, one colour per refinement."""
    import seaborn as sns

    series = list(dict.fromkeys(data['Refinement']))
    palette = sns.color_palette('Dark2', max(len(series), 3))
    for color, name in zip(palette, series):
        rows = data[data['Refinement'] == name].sort_values('Epoch')
        suffix = f' ({name})' if len(series) > 1 else ''
        for column, ls in (('Rwork', '-'), ('Rfree', '--')):
            ax.plot(rows['Epoch'], rows[column], color=color, ls=ls,
                    label=column + suffix, marker=_lone_point_marker(rows[column]))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R')
    ax.grid(which='both', axis='both', ls='-.')
    ax.legend(fontsize='small', framealpha=0.85)


def _draw_peaks(ax, peak_data):
    """Anomalous peak height against epoch: colour by atom type, line style by
    residue. A site with a single epoch so far -- every site after the first --
    also gets a marker, since a one-point line draws nothing."""
    import seaborn as sns

    # Explicit orders, so the markers below take the same colours as the lines.
    hue_order = sorted(peak_data['AtomType'].unique())
    style_order = list(dict.fromkeys(peak_data['Residue']))
    common = dict(x='Epoch', y='peakz', hue='AtomType', style='Residue',
                  palette='Dark2', hue_order=hue_order, style_order=style_order, ax=ax)
    sns.lineplot(peak_data, **common)
    epochs_seen = peak_data.groupby('Residue')['Epoch'].transform('nunique')
    lone = peak_data[epochs_seen == 1]
    if not lone.empty:
        sns.scatterplot(lone, legend=False, s=70, **common)


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
        # What the run is doing, from its `[status]` lines (abismal.status).
        self.status_label = widgets.HTML('')
        # Refinements by result directory name -> 'started' | 'finished' |
        # 'failed' | 'skipped', from the `[refinement]` lines.
        self._refinements = {}
        self.refinement_status = widgets.HTML('')
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
        # Beside the peak heights, and likewise hidden until a refinement has
        # finished; every refinement reports R values, anomalous or not.
        self.r_factors_widget = widgets.Output()
        self.r_factors_label = widgets.HTML(
            value="<b>R<sub>work</sub> / R<sub>free</sub></b>",
            layout=widgets.Layout(display='none'),
        )
        self._r_factors_signature = None
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
        cmd = ['abismal', *self.args, '--verbosity', '2']
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
            self.status_label,
            widgets.HTML("<b>Training History</b>"),
            self.history_widget,
            # Side by side where there is room, the peaks below the R values where
            # there is not.
            widgets.HBox([
                widgets.VBox([self.r_factors_label, self.r_factors_widget]),
                widgets.VBox([self.peaks_label, self.peaks_widget]),
            ], layout=widgets.Layout(flex_flow='row wrap')),
        ]
        if self.has_phenix:
            sections += [
                widgets.HTML("<b>Refinement Results</b>"),
                self.refinement_status,
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
        # Filtered here rather than at capture: console.log on disk keeps every
        # line, including the ones this hides, so nothing is lost. See
        # _log_filter for what counts as noise (currently nothing).
        if is_noise(line):
            return
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
                    (self.status_label, ['value']),
                    (self.refinement_status, ['value']),
                    (self.stop_button, ['disabled']),
                    (self.history_widget, ['outputs']),
                    (self.peaks_label.layout, ['display']),
                    (self.peaks_widget, ['outputs']),
                    (self.r_factors_label.layout, ['display']),
                    (self.r_factors_widget, ['outputs']),
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

    def _handle_line(self, line):
        """Pick the run's `[status]` and `[refinement]` lines out of its output."""
        m = _STATUS_RE.match(line)
        if m:
            text = f'<span style="color:#555">{html.escape(m.group(1))}</span>'
            self._run_on_main_thread(lambda: setattr(self.status_label, 'value', text))
            return
        m = _REFINEMENT_RE.match(line)
        if m:
            event, name = m.groups()
            # A skipped epoch only replaces nothing: it never ran.
            if event != 'skipped' or name not in self._refinements:
                self._refinements[name] = event
            self._render_refinements()

    def _render_refinements(self):
        """List the running refinements with their current stage, and a summary of
        the rest, under Refinement Results."""
        if not self._refinements:
            return
        by_epoch = sorted(self._refinements.items(), key=lambda kv: (_epoch_of(kv[0]) or 0, kv[0]))
        several = len({name.split('_asu_')[0] for name, _ in by_epoch}) > 1
        def label(name):
            epoch = _epoch_of(name)
            prefix = f'{name.split("_asu_")[0]}, ' if several else ''
            return f'{prefix}epoch {epoch}'
        lines = []
        for name, state in by_epoch:
            if state == 'started':
                stage = _refinement_stage(os.path.join(self.out_dir, name))
                lines.append(f'&#9654; <b>{html.escape(label(name))}</b>: {stage}')
        done = [label(n) for n, s in by_epoch if s == 'finished']
        failed = [n for n, s in by_epoch if s == 'failed']
        skipped = [label(n) for n, s in by_epoch if s == 'skipped']
        if done:
            lines.append(f'Finished: {len(done)} (latest {html.escape(done[-1])})')
        for name in failed:
            lines.append(
                f'<span style="color:#b00">Failed: {html.escape(label(name))} '
                f'(see {html.escape(os.path.join(name, "stderr.txt"))})</span>'
            )
        if skipped:
            lines.append(f'Skipped while another was running: {", ".join(map(html.escape, skipped))}')
        if not any(s == 'started' for _, s in by_epoch):
            lines.insert(0, 'No refinement running')
        value = '<div style="line-height:1.5">' + '<br>'.join(lines) + '</div>'
        self._run_on_main_thread(lambda: setattr(self.refinement_status, 'value', value))

    def _update_progress(self, cur, total):
        def apply():
            self.progress_widget.max = total
            self.progress_widget.value = cur - 1
            self.progress_label.value = f'Epoch {cur} / {total}'
        self._run_on_main_thread(apply)

    def _tail(self):
        """Stream console.log into log_widget and drive the progress bar."""
        epoch_re = re.compile(r'Epoch (\d+)/(\d+)')

        # Attaching to a running job can beat the child to creating its log, so
        # the file legitimately may not exist yet. Opening it unguarded killed
        # this thread with a FileNotFoundError nobody saw -- it is a daemon and
        # nothing joins it -- leaving a runner that looked attached but never
        # showed a line or moved the progress bar.
        while not os.path.exists(self.console_log):
            if not self.is_running:
                return
            time.sleep(0.5)

        with open(self.console_log, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    self._append_log(line)
                    self._handle_line(line)
                    m = epoch_re.match(line)
                    if m:
                        self._update_progress(int(m.group(1)), int(m.group(2)))
                elif self.is_running:
                    time.sleep(0.5)
                else:
                    # Process done — drain any lines written just before exit.
                    for line in f:
                        self._append_log(line)
                        self._handle_line(line)
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
                    self.status_label.value = ''
                self._run_on_main_thread(finish_ok)
            else:
                def finish_fail(rc=rc):
                    self.progress_widget.bar_style = 'danger'
                    self.progress_label.value = f'Failed (exit {rc})'
                    self.status_label.value = ''
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
            self._update_r_factors()
            # The stage of a running refinement changes without a line in our log.
            self._render_refinements()

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
            fig = Figure(figsize=(6, 3.6))
            ax = fig.subplots()
            _draw_peaks(ax, peak_data)
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            ax.grid(which='both', axis='both', ls='-.')
            ax.set_ylabel(r"Anomalous Peak Height ($\sigma$)")
            fig.tight_layout()

            self._show_figure(self.peaks_widget, fig)
        self._peaks_signature = signature
        self._run_on_main_thread(
            lambda: setattr(self.peaks_label.layout, 'display', '')
        )

    def _result_dirs(self):
        """Every refinement result directory under out_dir, either backend."""
        dirs = []
        for prefix in ("eff", "torchref"):
            dirs.extend(glob.glob(str(Path(self.out_dir) / f"{prefix}_*_asu_*_epoch_*")))
        return sorted(dirs)

    def _read_r_factors(self):
        """Rwork and Rfree per finished refinement, tagged with epoch and with the
        refinement ("torchref_0", "eff_1", ...) for runs that refine more than one
        model. None if there are none yet."""
        rows = []
        for d in self._result_dirs():
            name, _, epoch = Path(d).name.partition('_asu_')
            try:
                epoch = int(epoch.split('_epoch_')[-1])
            except ValueError:
                continue
            r = _refinement_r_factors(d)
            if r is not None:
                rows.append({'Refinement': name, 'Epoch': epoch, 'Rwork': r[0], 'Rfree': r[1]})
        return pd.DataFrame(rows) if rows else None

    def _update_r_factors(self):
        """Plot Rwork and Rfree against epoch beside the peak heights."""
        dirs = self._result_dirs()
        if not dirs:
            return
        # Redraw only when a result has appeared or changed, as for the peaks.
        signature = tuple(
            (f, _mtime(f)) for d in dirs
            for f in sorted(glob.glob(os.path.join(d, 'stdout.txt')) + glob.glob(os.path.join(d, '*.log')))
        )
        if signature == self._r_factors_signature:
            return
        data = self._read_r_factors()
        if data is None:
            return

        from matplotlib.figure import Figure

        with _FIGURE_LOCK:
            fig = Figure(figsize=(3.6, 3.6))
            _draw_r_factors(fig.subplots(), data)
            fig.tight_layout()
            self._show_figure(self.r_factors_widget, fig)
        self._r_factors_signature = signature
        self._run_on_main_thread(
            lambda: setattr(self.r_factors_label.layout, 'display', '')
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
            #
            # Not marked `isolated`: iframe_html is already sealed in its own srcdoc
            # iframe, and JupyterLab honours `isolated` by wrapping the output in a
            # second one. The reload broadcast below reaches only the iframes in the
            # notebook's own document, so with the extra frame it stopped at the
            # wrapper and the viewer stayed on its first epoch while the label
            # above it kept advancing.
            self.viewer_widget.outputs = ({
                'output_type': 'display_data',
                'data': {'text/html': viewer.iframe_html},
                'metadata': {},
            },)
            self._viewer_initialized = True
        else:
            # Ask the viewer to reload in place, preserving camera orientation.
            #
            # Broadcast to every iframe and let the viewers filter on the id in
            # the payload. Selecting the frame here instead means reading
            # f.contentWindow.ABISMAL_VIEWER_ID, and reading a property off
            # another origin's window throws -- which on Colab, where outputs are
            # sandboxed onto their own origin, is every frame. The old code
            # caught that per frame and returned false, so the search quietly
            # found nothing and the viewer sat on epoch 1 for the whole run.
            #
            # postMessage itself is fine cross-origin; it is designed for it.
            js = (
                f'/*{time.time()}*/(function(){{'
                f'var m={json.dumps(payload)};'
                f'var f=document.querySelectorAll("iframe");'
                f'for(var i=0;i<f.length;i++){{'
                f'try{{f[i].contentWindow.postMessage(m,"*");}}catch(e){{}}}}'
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
            # Refinements usually finish after training, so the charts follow too;
            # each redraws only when its files have changed.
            self._update_peaks()
            self._update_r_factors()
            if self._last_pdb == prev:
                unchanged += 1
            else:
                unchanged = 0
        # Nothing polls after this, so the Colab sync loop has no reason to keep
        # calling back into the kernel. _tail already does this on the branch with no
        # refinement; this is the same signal for the branch that has one.
        self._monitoring_active = False
