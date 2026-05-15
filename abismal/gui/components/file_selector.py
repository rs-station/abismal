import os
from pathlib import Path
from ipywidgets import widgets


def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


class ServerFileSelectorWidget(widgets.VBox):
    """
    Two-panel file selector: left panel navigates directories (single-click),
    right panel lists matching files with native shift/ctrl-click multi-select.
    Click "Add Selected Files" to accumulate files across directories.
    """

    header_string = "Select Files"

    def __init__(self, initial_directory=".", **kwargs):
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
                self.selected_files_label,
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

        self.selected_files_label = widgets.HTML(
            value=self._format_selected(),
            layout=widgets.Layout(
                max_height='250px',
                overflow_y='auto',
                border='1px solid lightgray',
                padding='4px',
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
        super().__init__()

    def file_filter(self, file_name):
        return any(file_name.endswith(s) for s in self.file_types)


class PhenixFileSelector(ServerFileSelectorWidget):
    header_string = "Configuration (*.eff) file for phenix.refine"
    file_types = [".eff"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def file_filter(self, file_name):
        return any(file_name.endswith(s) for s in self.file_types)

    @property
    def value(self):
        return ",".join(self.get_selected_files())


class ColabFileSelectorWidget(widgets.VBox):
    """File selector for Google Colab using ipywidgets.FileUpload."""

    header_string = "Select Files"
    accepted_extensions = ""
    upload_dir = "/tmp/abismal_uploads"

    def __init__(self, *args, **kwargs):
        self._saved_paths = []
        self._header = widgets.HTML(f"<h3>{self.header_string}</h3>")
        self._upload_widget = widgets.FileUpload(
            accept=self.accepted_extensions,
            multiple=True,
        )
        self._files_label = widgets.HTML("<p><i>No files uploaded</i></p>")
        self._upload_widget.observe(self._on_upload, names="value")
        super().__init__(
            children=[self._header, self._upload_widget, self._files_label],
            **kwargs,
        )

    def _on_upload(self, change):
        os.makedirs(self.upload_dir, exist_ok=True)
        self._saved_paths = []
        val = self._upload_widget.value
        # ipywidgets 7: dict of {filename: {'content': bytes, ...}}
        # ipywidgets 8: tuple of {'name': str, 'content': bytes, ...}
        if isinstance(val, dict):
            items = [(name, info["content"]) for name, info in val.items()]
        else:
            items = [(f["name"], f["content"]) for f in val]
        for name, content in items:
            path = os.path.join(self.upload_dir, name)
            with open(path, "wb") as f:
                f.write(content)
            self._saved_paths.append(path)
        if self._saved_paths:
            html = "<p><b>Uploaded:</b></p><ul>" + "".join(
                f"<li><code>{p}</code></li>" for p in self._saved_paths
            ) + "</ul>"
        else:
            html = "<p><i>No files uploaded</i></p>"
        self._files_label.value = html

    @property
    def value(self):
        return list(self._saved_paths)


class ColabReflectionFileSelector(ColabFileSelectorWidget):
    header_string = "Input Reflection Files (*.stream, *.expt/refl, *.mtz)"
    accepted_extensions = ".stream,.expt,.refl,.mtz,.json,.pickle"


class ColabPhenixFileSelector(ColabFileSelectorWidget):
    header_string = "Configuration (*.eff) file for phenix.refine"
    accepted_extensions = ".eff"

    @property
    def value(self):
        return ",".join(self._saved_paths)
