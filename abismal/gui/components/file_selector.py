import html
import os
from pathlib import Path
from ipywidgets import widgets


def _label(text):
    """The right-aligned label column shared by every control row."""
    return widgets.HTML(
        value=f'<div style="text-align:right;line-height:32px;padding-right:6px">{html.escape(text)}</div>',
        layout=widgets.Layout(width='120px', min_width='120px'),
    )


def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _running_servers():
    """Every jupyter server this user has running, or [] if we cannot ask."""
    try:
        from jupyter_server.serverapp import list_running_servers
        return [info for info in list_running_servers() if info]
    except Exception:
        # No jupyter_server, or a runtime directory we cannot read. Either way
        # there is nothing to learn here, and this runs at widget construction.
        return []


def _this_server(servers):
    """Whichever running server spawned this kernel, as best as can be told.

    ipykernel puts the spawning server's pid in JPY_PARENT_PID, which settles it
    outright. Failing that, fall back to the root the kernel's cwd sits under --
    the deepest one, since nested roots both match.
    """
    if not servers:
        return None

    parent = os.environ.get('JPY_PARENT_PID', '')
    if parent.isdigit():
        for info in servers:
            if info.get('pid') == int(parent):
                return info

    cwd = os.path.realpath(os.getcwd())
    containing = [
        info for info in servers
        if info.get('root_dir') and (
            cwd == os.path.realpath(info['root_dir'])
            or cwd.startswith(os.path.realpath(info['root_dir']) + os.sep)
        )
    ]
    if containing:
        return max(containing, key=lambda info: len(os.path.realpath(info['root_dir'])))
    return servers[0]


def _jupyter_launch_directory():
    """The directory ``jupyter lab`` was actually run from, or None.

    The server process does not chdir at startup, so its cwd *is* the directory
    it was launched from. That is not ``root_dir``: launching as ``jupyter lab
    some/dir/nb.ipynb`` sets root_dir to the notebook's own directory, and
    ``--ServerApp.root_dir`` sets it to wherever you say, while the process stays
    where it started. Nor is it the kernel's cwd, which jupyter sets to the
    notebook's directory.

    In descending order of how much it can be trusted:

    1. ``/proc/<server pid>/cwd`` -- the live answer, Linux only.
    2. ``PWD`` -- the shell's directory at launch, inherited by the server and
       through it by this kernel. Only consulted when JPY_PARENT_PID says we are
       under a server, since otherwise it is just our own cwd by another name.
       Missing if the server was started by something that does not set it.
    3. ``root_dir`` -- equal to the launch directory whenever ``jupyter lab`` was
       started with no file argument, and wrong when it was not. A guess.
    """
    info = _this_server(_running_servers())

    pids = []
    if info and info.get('pid'):
        pids.append(info['pid'])
    parent = os.environ.get('JPY_PARENT_PID', '')
    parent = int(parent) if parent.isdigit() else None
    if parent is not None and parent not in pids:
        pids.append(parent)

    for pid in pids:
        try:
            return os.path.realpath(os.readlink(f'/proc/{pid}/cwd'))
        except OSError:
            continue

    if parent is not None:
        pwd = os.environ.get('PWD')
        if pwd and os.path.isdir(pwd):
            return os.path.realpath(pwd)

    if info and info.get('root_dir'):
        return os.path.realpath(info['root_dir'])
    return None


def default_directory():
    """Where the file pickers open, and what relative paths resolve against.

    The directory jupyter was launched from, which is the one the user was
    standing in when they started -- not the kernel's cwd, which is wherever the
    .ipynb happens to sit and is the abismal checkout for anyone who opened the
    shipped notebook in place.

    Note this is deliberately *not* the server's ``root_dir``, which is what
    :func:`abismal.gui.runner._files_url` resolves against: launch jupyter on a
    notebook outside your data directory and the two differ, at which point the
    3D viewer cannot fetch results from here. Keeping them together is the
    launch instruction in the README, not something this can fix.

    Falls back to /content on Colab and to the cwd anywhere else.
    """
    launched_from = _jupyter_launch_directory()
    if launched_from:
        return launched_from

    if _is_colab() and os.path.isdir('/content'):
        return '/content'
    return os.getcwd()


class ServerFileSelectorWidget(widgets.VBox):
    """
    Two-panel file selector: left panel navigates directories (single-click),
    right panel lists matching files with native shift/ctrl-click multi-select.
    Click "Add Selected Files" to accumulate files across directories.
    """

    header_string = "Select Files"

    def __init__(self, initial_directory=None, **kwargs):
        if initial_directory is None:
            initial_directory = default_directory()
        self._current_dir = Path(initial_directory).resolve()
        self._selected_files = []
        self._create_widgets()
        super().__init__(
            children=[
                self.header_label,
                self.nav_bar,
                self.browser_box,
                self.nav_bar2,
                self.add_button,
                self.selected_files_box,
                self.clear_button,
            ],
            **kwargs,
        )
        self._refresh()

    def file_filter(self, file_name):
        return True

    def _create_widgets(self):
        self.header_label = widgets.HTML(value=f"<h3>{self.header_string}</h3>")

        self.dir_label = widgets.Label(
            value=str(self._current_dir),
            layout=widgets.Layout(flex='1'),
        )
        self.status_label = widgets.Label(value='')
        self.up_button = widgets.Button(
            description='↑ Up',
            layout=widgets.Layout(width='80px'),
        )
        self.up_button.on_click(self._go_up)
        self.nav_bar = widgets.HBox([self.dir_label, self.up_button])

        self.dir_list = widgets.Select(
            options=[],
            layout=widgets.Layout(width='100%', height='250px'),
        )
        self.dir_list.observe(self._on_dir_select, names='value')

        self.file_list = widgets.SelectMultiple(
            options=[],
            layout=widgets.Layout(width='100%', height='250px'),
        )

        dir_col = widgets.VBox(
            [widgets.HTML('<b>Directories</b>'), self.dir_list],
            layout=widgets.Layout(flex='1'),
        )
        file_col = widgets.VBox(
            [widgets.HTML('<b>Files</b> (shift-click or ctrl-click to multi-select)'), self.file_list],
            layout=widgets.Layout(flex='3'),
        )
        self.browser_box = widgets.HBox(
            [dir_col, file_col],
            layout=widgets.Layout(width='100%'),
        )
        self.nav_bar2 = widgets.HBox([self.status_label])

        self.add_button = widgets.Button(
            description='Add Selected Files',
            button_style='primary',
            icon='plus',
        )
        self.add_button.on_click(self._on_add_clicked)

        self.selected_files_label = widgets.HTML(value=self._format_selected())
        self.selected_files_box = widgets.Box(
            [self.selected_files_label],
            layout=widgets.Layout(
                max_height='250px',
                overflow_y='auto',
                border='1px solid lightgray',
                padding='4px',
                display='block',
                width='100%',
            ),
        )

        self.clear_button = widgets.Button(
            description='Clear Selection',
            button_style='warning',
            icon='times',
        )
        self.clear_button.on_click(self._on_clear_clicked)

    def _refresh(self):
        try:
            entries = list(os.scandir(self._current_dir))
        except PermissionError:
            return

        dirs = sorted(
            [e.name for e in entries if e.is_dir() and not e.name.startswith('.')],
            key=str.lower,
        )
        files = sorted(
            [e.name for e in entries if not e.is_dir() and self.file_filter(e.name)],
            key=str.lower,
        )

        self.dir_label.value = str(self._current_dir)
        self.status_label.value = f'{len(files)} matching file(s) found'

        # Detach observer while mutating options to prevent spurious navigation.
        try:
            self.dir_list.unobserve(self._on_dir_select, names='value')
        except ValueError:
            pass
        self.dir_list.options = dirs
        # Reset value so clicking any entry (including the first) always fires observe.
        self.dir_list.value = None
        self.dir_list.observe(self._on_dir_select, names='value')

        self.file_list.options = tuple(files)
        self.file_list.value = ()

    def _go_up(self, _):
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._current_dir = parent
            self._refresh()

    def _on_dir_select(self, change):
        name = change['new']
        if not name:
            return
        target = self._current_dir / name
        if target.is_dir() and os.access(str(target), os.R_OK):
            self._current_dir = target
            self._refresh()

    def _on_add_clicked(self, _):
        for fname in self.file_list.value:
            path = str(self._current_dir / fname)
            if path not in self._selected_files:
                self._selected_files.append(path)
        self._update_selected_label()

    def _on_clear_clicked(self, _):
        self._selected_files = []
        self._update_selected_label()

    def _format_selected(self):
        if not self._selected_files:
            return "<p><i>No files selected</i></p>"
        items = "".join(f"<li><code>{p}</code></li>" for p in self._selected_files)
        n = len(self._selected_files)
        return f"<p><b>{n} file(s) selected:</b></p><ul>{items}</ul>"

    def _update_selected_label(self):
        self.selected_files_label.value = self._format_selected()

    def get_selected_files(self):
        return list(self._selected_files)

    def set_directory(self, directory):
        self._current_dir = Path(directory).resolve()
        self._refresh()

    @property
    def value(self):
        return self.get_selected_files()


class ReflectionFileSelector(ServerFileSelectorWidget):
    header_string = "Input Reflection Files (*.stream, *.expt/refl, *.mtz)"
    file_types = [".mtz", ".expt", ".refl", ".json", ".pickle", ".stream"]

    def __init__(self, *args, **kwargs):
        kwargs.pop('name', None)
        kwargs.pop('action', None)
        super().__init__(**kwargs)

    def file_filter(self, file_name):
        return any(file_name.endswith(s) for s in self.file_types)


class PathSelector(widgets.VBox):
    """A text field with a Browse button that expands a file browser inline.

    The two-panel :class:`ServerFileSelectorWidget` is the right shape for
    ``inputs``, where you accumulate a list across directories, but it is 250px
    tall and there are seven path-valued arguments in the parser. This is the
    compact form: one row that looks like every other labelled control, and a
    browser that appears underneath only while you are using it.

    Typing a path still works -- the field is the value, and browsing is just a
    way to fill it in -- so nothing that worked before the browser existed
    stops working.

    mode:
        ``'file'``      one file; Browse fills the field with its path.
        ``'files'``     several files, comma-joined, matching the CLI's
                        comma-separated ``--eff-files``/``--torchref-pdb``.
        ``'directory'`` no file list; Browse takes the directory you navigate to.
        ``'save'``      a directory that need not exist yet. Browse picks the
                        parent, a second field holds the name, and a status line
                        says what will happen. See the class note below.

    Why ``'save'`` is a mode of its own: the other three are *open* pickers,
    where a listing of what exists is the complete set of valid answers.
    ``--out-dir`` is a save target -- the directory you want usually does not
    exist -- so browsing alone can never express it. This is the shape every
    Save As dialog uses, and nothing is created until the job is launched;
    :meth:`AbismalRunner.start` already makes the directory.
    """

    def __init__(self, description='', mode='file', file_types=None,
                 value='', placeholder='', initial_directory=None,
                 tooltip='', name='', **kwargs):
        if mode not in ('file', 'files', 'directory', 'save'):
            raise ValueError(f"unknown mode {mode!r}")
        if initial_directory is None:
            initial_directory = default_directory()
        self.mode = mode
        self.file_types = tuple(file_types) if file_types else ()
        self._label_text = description
        self._current_dir = Path(initial_directory).resolve()

        self._label = _label(description)
        self._input = widgets.Text(
            value=value,
            placeholder=placeholder,
            layout=widgets.Layout(flex='1'),
        )
        self._browse_button = widgets.Button(
            description='Browse',
            tooltip=tooltip or f'Browse for {description}',
            layout=widgets.Layout(width='90px'),
        )
        self._browse_button.on_click(self._toggle_browser)
        row = widgets.HBox([self._label, self._input, self._browse_button])
        rows = [row]

        self._spacers = []
        self._name_input = None
        self._status = None
        if mode == 'save':
            self._name_input = widgets.Text(
                value=name,
                placeholder='new directory name',
                layout=widgets.Layout(flex='1'),
            )
            separator = widgets.HTML(
                value='<div style="line-height:32px;padding:0 6px">/</div>',
                layout=widgets.Layout(width='16px'),
            )
            name_spacer = _label('')
            self._spacers.append(name_spacer)
            rows.append(widgets.HBox([name_spacer, separator, self._name_input]))

            self._status = widgets.HTML(value='')
            status_spacer = _label('')
            self._spacers.append(status_spacer)
            rows.append(widgets.HBox([status_spacer, self._status]))

            self._input.observe(self._refresh_status, names='value')
            self._name_input.observe(self._refresh_status, names='value')
            self._refresh_status()

        self._browser = self._build_browser()
        self._browser.layout.display = 'none'
        rows.append(self._browser)
        super().__init__(rows, **kwargs)

    # The label is nested a level deeper than in the plain Text/Dropdown rows,
    # so hand _set_label_widths the widget directly rather than let it guess.
    @property
    def _label_widget(self):
        return self._label

    def _build_browser(self):
        self._dir_label = widgets.Label(
            value=str(self._current_dir), layout=widgets.Layout(flex='1'),
        )
        self._up_button = widgets.Button(
            description='↑ Up', layout=widgets.Layout(width='80px'),
        )
        self._up_button.on_click(self._go_up)

        self._dir_list = widgets.Select(
            options=[], layout=widgets.Layout(width='100%', height='160px'),
        )
        self._dir_list.observe(self._on_dir_select, names='value')
        columns = [widgets.VBox(
            [widgets.HTML('<b>Directories</b>'), self._dir_list],
            layout=widgets.Layout(flex='1'),
        )]

        if self.mode in ('directory', 'save'):
            self._file_list = None
        else:
            select_cls = (
                widgets.SelectMultiple if self.mode == 'files' else widgets.Select
            )
            hint = (
                '<b>Files</b> (shift-click or ctrl-click to multi-select)'
                if self.mode == 'files' else '<b>Files</b>'
            )
            self._file_list = select_cls(
                options=[], layout=widgets.Layout(width='100%', height='160px'),
            )
            columns.append(widgets.VBox(
                [widgets.HTML(hint), self._file_list],
                layout=widgets.Layout(flex='2'),
            ))

        self._select_button = widgets.Button(
            description='Select', button_style='primary', icon='check',
            layout=widgets.Layout(width='110px'),
        )
        self._select_button.on_click(self._on_select)
        self._cancel_button = widgets.Button(
            description='Cancel', layout=widgets.Layout(width='110px'),
        )
        self._cancel_button.on_click(self._close_browser)

        return widgets.VBox(
            [
                widgets.HBox([self._dir_label, self._up_button]),
                widgets.HBox(columns, layout=widgets.Layout(width='100%')),
                widgets.HBox([self._select_button, self._cancel_button]),
            ],
            layout=widgets.Layout(
                border='1px solid lightgray', padding='4px',
                margin='0 0 4px 0', width='100%',
            ),
        )

    def file_filter(self, file_name):
        if not self.file_types:
            return True
        return any(file_name.endswith(s) for s in self.file_types)

    def _refresh(self):
        try:
            entries = list(os.scandir(self._current_dir))
        except OSError:
            return
        self._dir_label.value = str(self._current_dir)

        dirs = sorted(
            [e.name for e in entries if e.is_dir() and not e.name.startswith('.')],
            key=str.lower,
        )
        # Detach while mutating options, or setting them fires the navigation
        # observer -- the same reason ServerFileSelectorWidget does this.
        try:
            self._dir_list.unobserve(self._on_dir_select, names='value')
        except ValueError:
            pass
        self._dir_list.options = dirs
        self._dir_list.value = None
        self._dir_list.observe(self._on_dir_select, names='value')

        if self._file_list is not None:
            files = sorted(
                [e.name for e in entries
                 if not e.is_dir() and self.file_filter(e.name)],
                key=str.lower,
            )
            self._file_list.options = tuple(files)
            self._file_list.value = () if self.mode == 'files' else None

    def _toggle_browser(self, _=None):
        if self.browser_open:
            self._close_browser()
        else:
            self._open_browser()

    @property
    def browser_open(self):
        return self._browser.layout.display != 'none'

    def _open_browser(self):
        # Reopen where the current value points, so correcting a typo does not
        # start over from the root.
        current = self._input.value.split(',')[0].strip()
        if current:
            candidate = Path(current).expanduser()
            if not candidate.is_absolute():
                candidate = Path(self._current_dir) / candidate
            if candidate.is_dir():
                self._current_dir = candidate.resolve()
            elif candidate.parent.is_dir():
                self._current_dir = candidate.parent.resolve()
        self._refresh()
        self._browser.layout.display = ''
        self._browse_button.description = 'Close'

    def _close_browser(self, _=None):
        self._browser.layout.display = 'none'
        self._browse_button.description = 'Browse'

    def _go_up(self, _):
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._current_dir = parent
            self._refresh()

    def _on_dir_select(self, change):
        name = change['new']
        if not name:
            return
        target = self._current_dir / name
        if target.is_dir() and os.access(str(target), os.R_OK):
            self._current_dir = target
            self._refresh()

    def _on_select(self, _):
        if self.mode in ('directory', 'save'):
            self._input.value = str(self._current_dir)
        elif self.mode == 'files':
            chosen = self._file_list.value or ()
            if not chosen:
                return
            self._input.value = ','.join(
                str(self._current_dir / name) for name in chosen
            )
        else:
            chosen = self._file_list.value
            if not chosen:
                return
            self._input.value = str(self._current_dir / chosen)
        self._close_browser()

    @property
    def value(self):
        if self.mode != 'save':
            return self._input.value
        parent = self._input.value.strip()
        name = self._name_input.value.strip().strip('/')
        if not name:
            return parent
        if not parent:
            return name
        return os.path.join(parent, name)

    def set_label_width(self, width):
        """Widen the label column, keeping the extra rows aligned under it.

        _set_label_widths sizes one label per control. Save mode has three rows,
        and the two without a label still have to line up with the one that has.
        """
        self._label.layout.width = width
        self._label.layout.min_width = width
        for spacer in self._spacers:
            spacer.layout.width = width
            spacer.layout.min_width = width

    def _refresh_status(self, _=None):
        """Say what launching would do: create, reuse, overwrite, or fail.

        The mistyped-parent case is the one worth catching. It sails through the
        form today and only fails inside the child process, where the message
        lands in console.log rather than anywhere the user is looking.
        """
        from abismal.gui.cleanup import find_abismal_outputs

        target = self.value.strip()
        if not target:
            self._set_status('', '')
            return

        path = Path(target).expanduser()
        if path.is_file():
            self._set_status('#b00', f'{path} is a file')
            return
        if path.is_dir():
            existing = find_abismal_outputs(str(path))
            if existing:
                self._set_status(
                    '#b26a00',
                    f'exists &mdash; {len(existing)} abismal output(s); '
                    'Run will offer to overwrite',
                )
            else:
                self._set_status('#555', 'exists, and holds no abismal output')
            return

        parent = path.parent
        if parent.is_dir():
            self._set_status('#0a0', 'will be created')
        else:
            self._set_status('#b00', f'parent {parent} does not exist')

    def _set_status(self, color, message):
        if not message:
            self._status.value = ''
            return
        self._status.value = (
            f'<div style="color:{color};line-height:24px;font-size:90%">'
            f'{message}</div>'
        )

    def set_directory(self, directory):
        self._current_dir = Path(directory).resolve()
        if self.browser_open:
            self._refresh()
