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
import pandas as pd
import matplotlib.pyplot as plt


def _jupyter_server_root_dir(file_path):
    """Return the JupyterLab server root_dir that contains file_path, or None.

    On OOD/JupyterLab the server's root_dir often differs from the kernel cwd,
    and /files/ URLs resolve relative to root_dir — so paths for embedded
    iframes must be computed relative to it, not to os.getcwd().
    """
    try:
        from jupyter_server.serverapp import list_running_servers
    except ImportError:
        return None
    abs_file = os.path.abspath(file_path)
    matches = []
    for info in list_running_servers():
        root = info.get('root_dir')
        if not root:
            continue
        root_abs = os.path.abspath(root)
        if abs_file == root_abs or abs_file.startswith(root_abs + os.sep):
            matches.append(root_abs)
    if not matches:
        return None
    return max(matches, key=len)


def _file_url_path(path):
    """Return a path usable under /files/<...> in JupyterLab."""
    abs_path = os.path.abspath(path)
    root = _jupyter_server_root_dir(abs_path)
    if root is not None:
        return os.path.relpath(abs_path, root)
    return os.path.relpath(abs_path)


class AbismalRunner:
    """
    Runs abismal as a detached subprocess with live output widgets.

    Output (stdout+stderr) is written to {out_dir}/console.log so it
    survives a kernel restart and can be reconnected via attach().
    """

    poll_interval = 10.0  # seconds

    def __init__(self, args, out_dir, has_phenix=False, total_epochs=None):
        self.args = args
        self.out_dir = str(out_dir)
        self.has_phenix = has_phenix
        self._pid = None
        self._process = None
        self._tailer_thread = None
        self._poll_timer = None
        self._last_pdb = None
        self._last_mtz = None

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Launch a fresh abismal subprocess."""
        os.makedirs(self.out_dir, exist_ok=True)
        cmd = ['abismal', *self.args, '--keras-verbosity', '2']
        fout = open(self.console_log, 'w')
        self._process = subprocess.Popen(
            cmd,
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
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

    @property
    def is_running(self):
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

    def _append_log(self, line):
        self._log_text += line
        self.log_widget.value = self._render_log_html()

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
        self._tailer_thread = threading.Thread(target=self._tail, daemon=True)
        self._tailer_thread.start()
        self._schedule_poll()

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
                        cur, total = int(m.group(1)), int(m.group(2))
                        self.progress_widget.max = total
                        self.progress_widget.value = cur - 1
                        self.progress_label.value = f'Epoch {cur} / {total}'
                elif self.is_running:
                    time.sleep(0.5)
                else:
                    # Process done — drain any lines written just before exit.
                    for line in f:
                        self._append_log(line)
                        m = epoch_re.match(line)
                        if m:
                            cur, total = int(m.group(1)), int(m.group(2))
                            self.progress_widget.max = total
                            self.progress_widget.value = cur - 1
                            self.progress_label.value = f'Epoch {cur} / {total}'
                    break

        self.stop_button.disabled = True

        if self._process is not None:
            rc = self._process.wait()
            if rc == 0:
                self.progress_widget.value = self.progress_widget.max
                self.progress_widget.bar_style = 'success'
                self.progress_label.value = 'Finished'
            else:
                self.progress_widget.bar_style = 'danger'
                self.progress_label.value = f'Failed (exit {rc})'
        else:
            self.progress_widget.bar_style = ''
            self.progress_label.value = 'Process exited (check abismal.log)'

        try:
            os.remove(self.pid_file)
        except OSError:
            pass

        self._poll()

        if self.has_phenix:
            threading.Thread(
                target=self._post_training_phenix_watcher, daemon=True
            ).start()

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

        loss_cols = [c for c in df.columns if 'loss' in c.lower()]
        cc_cols = [c for c in df.columns if 'CC' in c]
        n_plots = bool(loss_cols) + bool(cc_cols)
        if not n_plots:
            return

        from matplotlib.figure import Figure
        fig = Figure(figsize=(5 * n_plots, 4))
        axes = fig.subplots(1, n_plots)
        if n_plots == 1:
            axes = [axes]
        ax_idx = 0

        if loss_cols:
            ax = axes[ax_idx]; ax_idx += 1
            for col in loss_cols:
                ax.plot(df['Epoch'], df[col], label=col)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss')
            ax.legend()

        if cc_cols:
            ax = axes[ax_idx]; ax_idx += 1
            for col in cc_cols:
                ax.plot(df['Epoch'], df[col], label=col)
            ax.set_xlabel('Epoch')
            ax.set_title('CC')
            ax.legend()

        fig.tight_layout()
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        self.history_widget.outputs = ({
            'output_type': 'display_data',
            'data': {'image/png': png_b64},
            'metadata': {},
        },)

    def _find_latest_phenix_results(self, asu_id=0):
        pattern = str(Path(self.out_dir) / f"eff_*_asu_{asu_id}_epoch_*")
        dirs = glob.glob(pattern)
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
            pdb_rel = _file_url_path(pdb_file)
            mtz_rel = _file_url_path(mtz_file)
        except ValueError:
            pdb_rel, mtz_rel = pdb_file, mtz_file

        try:
            from abismal.gui.components.uglymol import UglyMolViewer
            viewer = UglyMolViewer(
                pdb_file=pdb_rel, mtz_file=mtz_rel, viewer_id=self._viewer_id
            )
            map_keys = viewer.map_keys
        except Exception:
            # File is still being written by Phenix — retry on next poll.
            return

        # Advance cache only on success so a partial write doesn't block retries.
        self._last_pdb = pdb_file
        self._last_mtz = mtz_file

        epoch = Path(pdb_file).parent.name.split('_epoch_')[-1]
        self.viewer_label.value = (
            f'<b>Epoch {epoch}</b> &nbsp;|&nbsp; '
            f'<code>{Path(pdb_rel).name}</code> &nbsp;+&nbsp; <code>{Path(mtz_rel).name}</code>'
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
                f'if(t)t.contentWindow.postMessage({{'
                f'type:"reload",'
                f'pdb_file:"/files/{pdb_rel}",'
                f'mtz_file:"/files/{mtz_rel}",'
                f'map_keys:{json.dumps(map_keys)}'
                f'}},"*");'
                f'}})();'
            )
            self._js_widget.outputs = ({
                'output_type': 'display_data',
                'data': {'application/javascript': js},
                'metadata': {},
            },)

    def _post_training_phenix_watcher(self, max_unchanged=12):
        """Keep polling for Phenix results after training exits (Phenix may still be running)."""
        unchanged = 0
        while unchanged < max_unchanged:
            time.sleep(self.poll_interval)
            prev = self._last_pdb
            self._update_viewer()
            if self._last_pdb == prev:
                unchanged += 1
            else:
                unchanged = 0
