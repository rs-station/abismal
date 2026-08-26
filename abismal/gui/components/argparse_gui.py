from abismal.command_line.parser import parser as abismal_parser
from abismal.command_line.parser.custom_types import directory, list_of_paths
import argparse
from pathlib import Path
from ipywidgets import widgets
import time
from abismal.gui.components.file_selector import (
    PathSelector,
    ReflectionFileSelector,
    _label,
    default_directory,
)


def _set_label_widths(widget_list):
    """Set label column width for a group of widgets based on the longest label text."""
    labelable = [w for w in widget_list if hasattr(w, '_label_text')]
    if not labelable:
        return
    max_chars = max(len(w._label_text) for w in labelable)
    width = f'{max_chars * 8 + 24}px'
    for w in labelable:
        setter = getattr(w, 'set_label_width', None)
        if setter is not None:
            setter(width)
            continue
        lbl = getattr(w, '_label_widget', None) or w.children[0]
        lbl.layout.width = width
        lbl.layout.min_width = width


class _ToggleRow(widgets.HBox):
    _label_text = ''  # spacer only — button carries its own description

    def __init__(self, btn):
        super().__init__([_label(''), btn])
        self._btn = btn

    @property
    def value(self):
        return self._btn.value


class Text(widgets.HBox):
    def __init__(self, **kwargs):
        description = kwargs.pop("description", "")
        self._label_text = description
        self._input = widgets.Text(layout=widgets.Layout(flex='1'), **kwargs)
        super().__init__([_label(description), self._input])

    @property
    def value(self):
        return self._input.value


class Dropdown(widgets.HBox):
    def __init__(self, **kwargs):
        description = kwargs.pop("description", "")
        self._label_text = description
        self._dropdown = widgets.Dropdown(layout=widgets.Layout(flex='1'), **kwargs)
        super().__init__([_label(description), self._dropdown])

    @property
    def value(self):
        return self._dropdown.value


class ArgparseGUIBase:
    custom_widgets = {}
    custom_actions = {}

    # Any argument naming something on disk carries one of these types, and gets
    # a browsable field instead of a bare text box. Adding a path argument to the
    # parser is therefore all it takes to get a picker for it -- there is no
    # second list here to keep in sync.
    path_modes = {
        Path: 'file',
        list_of_paths: 'files',
        # A directory argument in this parser is always somewhere to write, and
        # the directory you want usually does not exist yet -- so it gets the
        # save-as row rather than a picker over what happens to be there.
        directory: 'save',
    }

    # Which suffixes each picker offers. Absent means "show every file"; the
    # field still accepts anything typed into it either way.
    path_file_types = {
        'reference_mtz': ('.mtz',),
        'r_free_mtz': ('.mtz',),
        'eff_files': ('.eff',),
        'torchref_pdb': ('.pdb', '.cif'),
        'posterior_init_file': ('.keras',),
        'scale_init_file': ('.keras',),
    }
    skipped_actions = [
        "help",
        "list_devices",
        "run_eagerly",
        "debug",
        "embed",
        "keras_verbosity",
    ]

    def _make_group_container(self, named_children):
        """Build a tabbed container for the per-group panels.

        Implemented as a row of buttons that toggle each panel's visibility
        rather than widgets.Tab/Accordion, which don't render under Colab's
        custom widget manager + legacy ipywidgets bundle. Buttons and Box
        visibility toggling work on both Colab and JupyterLab.
        """
        titles = list(named_children.keys())
        panels = list(named_children.values())
        buttons = []

        def select(idx):
            for i, panel in enumerate(panels):
                panel.layout.display = '' if i == idx else 'none'
            for i, btn in enumerate(buttons):
                btn.button_style = 'primary' if i == idx else ''

        for i, title in enumerate(titles):
            btn = widgets.Button(description=title)
            btn.on_click(lambda _b, idx=i: select(idx))
            buttons.append(btn)

        tab_bar = widgets.HBox(buttons)
        content = widgets.VBox(panels)
        select(0)
        return widgets.VBox([tab_bar, content])

    def __init__(self, parser=None):
        self.parser = parser if parser is not None else abismal_parser
        # The runner whose widget the form is currently showing, so that the next Run
        # click can put it down. See _install_runner.
        self._runner = None

    @staticmethod
    def action_to_name(action):
        if action.metavar is not None:
            return action.metavar
        return action.dest

    def to_args(self):
        args = []
        for action, widget in self._all_args.items():
            v = widget.value
            if v == "" or v == [] or v is None:
                continue
            if isinstance(action, argparse._StoreTrueAction):
                if not v:
                    continue
                args.append(action.option_strings[0])
            elif isinstance(action, argparse._StoreFalseAction):
                if v:
                    continue
                args.append(action.option_strings[0])
            else:
                if action.option_strings:
                    args.append(action.option_strings[0])
                if isinstance(v, list):
                    args.extend(v)
                else:
                    args.append(v)
        return list(map(str, args))

    def to_parser(self):
        return self.parser.parse_args(self.to_args())

    def action_to_widget(self, action, name=None):
        if name is None:
            name = self.action_to_name(action)
        if name in self.custom_widgets:
            return self.custom_widgets[name](action, name=name)
        if isinstance(action, argparse._StoreTrueAction):
            return _ToggleRow(widgets.ToggleButton(
                value=False,
                description=name,
                disabled=False,
                button_style="",
                tooltip=action.help or "",
            ))
        if isinstance(action, argparse._StoreFalseAction):
            return _ToggleRow(widgets.ToggleButton(
                value=True,
                description=name,
                disabled=False,
                button_style="",
                tooltip=action.help or "",
            ))
        mode = self.path_modes.get(action.type)
        if mode is not None:
            return PathSelector(
                description=name,
                mode=mode,
                file_types=self.path_file_types.get(action.dest),
                tooltip=action.help or "",
                **self._path_default(action, mode),
            )
        if isinstance(action, argparse._StoreAction):
            if action.choices is not None:
                default = action.default
                if action.type is not None and default is not None:
                    default = action.type(default)
                return Dropdown(
                    options=action.choices,
                    value=default,
                    description=name,
                    disabled=False,
                )
            else:
                placeholder = str(action.default) if action.default is not None else ""
                return Text(
                    placeholder=placeholder,
                    description=name,
                    tooltip=action.help or "",
                )
        return None

    @staticmethod
    def new_run_name():
        """The name offered for a fresh output directory.

        Timestamped so that re-running does not land on the previous run and
        raise the overwrite dialog every time. It is fixed when the form is
        built, so a notebook left open overnight offers yesterday's stamp until
        the cell is re-run -- still unique against the last run, and editable.
        """
        return time.strftime('abismal_%Y-%m-%d_%H%M')

    @classmethod
    def _path_default(cls, action, mode):
        """Prefill or placeholder text for a path field, from the CLI default.

        A relative default -- out_dir's "." -- means "right here", which on the
        command line is the cwd but in the notebook has to be the base directory
        the pickers use, or output lands next to the .ipynb. Resolve it and put
        it in the field as a real value, both so it is visible and so the child
        process is handed an absolute path rather than inheriting the kernel's
        idea of where "." is.
        """
        default = action.default
        if mode == 'save':
            base = Path(default_directory())
            if isinstance(default, Path) and default.is_absolute():
                base = default
            elif isinstance(default, Path):
                base = (base / default).resolve()
            return {"value": str(base), "name": cls.new_run_name()}
        if isinstance(default, Path):
            if default.is_absolute():
                return {"placeholder": str(default)}
            base = Path(default_directory())
            return {"value": str((base / default).resolve())}
        if default is not None:
            return {"placeholder": str(default)}
        return {}

    @staticmethod
    def _resolved_out_dir(parsed):
        """parsed.out_dir as an absolute path.

        The kernel and the child disagree about "." -- the child is launched with
        cwd=default_directory(), the kernel's cwd is the notebook's directory --
        and the runner opens console.log itself, so a relative out_dir would have
        it watching a different directory than the one being written to.
        """
        return str((Path(default_directory()) / Path(parsed.out_dir)).resolve())

    def run_abismal(self, button=None):
        from abismal.gui.runner import AbismalRunner
        from abismal.gui.cleanup import find_abismal_outputs
        parsed = self.to_parser()
        out_dir = self._resolved_out_dir(parsed)
        # "has_phenix" gates the refinement results viewer; either phenix or
        # torchref refinement produces per-epoch pdb+mtz results to display.
        has_phenix = (
            getattr(parsed, "eff_files", None) is not None
            or getattr(parsed, "torchref_pdb", None) is not None
        )

        runner = AbismalRunner.attach(out_dir, has_phenix=has_phenix)
        if runner is not None:
            runner._append_log(
                f'[Reconnected to running process PID {runner._pid}]\n'
            )
            runner.resume()
            self._install_runner(runner)
            return

        existing = find_abismal_outputs(out_dir)
        if existing:
            self._show_overwrite_confirm(out_dir, existing, parsed, has_phenix)
            return

        self._launch_runner(parsed, has_phenix)

    def _launch_runner(self, parsed, has_phenix):
        from abismal.gui.runner import AbismalRunner
        out_dir = self._resolved_out_dir(parsed)
        args = self.to_args()
        runner = AbismalRunner(
            args, out_dir, has_phenix=has_phenix,
            total_epochs=parsed.epochs,
            cwd=default_directory(),
        )
        runner.start()
        self._install_runner(runner)

    def _install_runner(self, runner):
        """Show `runner`'s widget in place of whatever ran before it.

        Swapping the widget out is not enough to stop the runner behind it: it keeps a
        self-rescheduling poll timer, a phenix watcher thread, and on Colab a browser
        interval that goes on syncing its widgets. So every Run click used to leave
        another runner polling out_dir forever -- reading a history.csv and a set of
        per-epoch result directories that the next run is about to delete and rewrite.
        """
        previous, self._runner = self._runner, runner
        if previous is not None and previous is not runner:
            previous.shutdown()
        self.widget.children = (
            self.top_section, self.tab, self.run_button,
            self._run_output, runner.to_widget(),
        )

    def _stop_runner(self):
        """Put the current runner down, leaving its subprocess alone."""
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.shutdown()

    def _show_overwrite_confirm(self, out_dir, existing, parsed, has_phenix):
        from abismal.gui.cleanup import cleanup_abismal_outputs

        warning = widgets.HTML(
            '<div style="color:#b00;font-weight:bold;margin-bottom:4px;">'
            f'Output directory <code>{out_dir}</code> already contains abismal '
            'outputs. The following will be removed:'
            '</div>'
        )
        file_list_html = widgets.HTML(
            value='<ul style="margin:0;">' + ''.join(
                f'<li><code>{p}</code></li>' for p in existing
            ) + '</ul>',
        )
        file_list_box = widgets.Box(
            [file_list_html],
            layout=widgets.Layout(
                max_height='250px',
                overflow_y='auto',
                border='1px solid lightgray',
                padding='4px',
                display='block',
                width='100%',
            ),
        )
        overwrite_btn = widgets.Button(
            description='Overwrite and Run',
            button_style='danger',
            icon='trash',
        )
        cancel_btn = widgets.Button(
            description='Cancel',
        )
        button_row = widgets.HBox([overwrite_btn, cancel_btn])
        confirm_box = widgets.VBox([warning, file_list_box, button_row])

        normal_children = (
            self.top_section, self.tab, self.run_button, self._run_output,
        )

        def _on_overwrite(_):
            # Before the files go, not after: the previous runner polls out_dir on a
            # timer, and a poll that overlaps the delete hands the 3D viewer a result
            # directory that is gone by the time the browser fetches it, which leaves
            # the viewer stuck on "Loading..." with nothing left to retry.
            self._stop_runner()
            cleanup_abismal_outputs(out_dir)
            self._launch_runner(parsed, has_phenix)

        def _on_cancel(_):
            self.widget.children = normal_children

        overwrite_btn.on_click(_on_overwrite)
        cancel_btn.on_click(_on_cancel)

        self.widget.children = normal_children + (confirm_box,)

    def to_widget(self):
        self.run_button = widgets.Button(
            description="Run Abismal",
            tooltip="Run Abismal merging",
        )
        # Capture exceptions from the click handler — on Colab they otherwise
        # go nowhere and the click silently does nothing.
        self._run_output = widgets.Output()

        def _on_run_click(button):
            import traceback
            self._run_output.clear_output()
            with self._run_output:
                try:
                    self.run_abismal(button)
                except SystemExit:
                    # argparse calls sys.exit() on validation failure; the
                    # usage/error message is already in the Output widget.
                    pass
                except Exception:
                    traceback.print_exc()
        self.run_button.on_click(_on_run_click)
        top_widgets = []
        tab_widgets = {}
        self._all_args = {}

        for group in self.parser._action_groups:
            for action in group._group_actions:
                name = self.action_to_name(action)
                if name in self.skipped_actions:
                    continue
                if name in self.custom_actions:
                    widget = self.custom_actions[name](action)
                else:
                    widget = self.action_to_widget(action)
                if widget is None:
                    continue
                self._all_args[action] = widget
                if action.required:
                    top_widgets.append(widget)
                else:
                    group_name = group.title
                    if group_name not in tab_widgets:
                        tab_widgets[group_name] = []
                    tab_widgets[group_name].append(widget)

        _set_label_widths(top_widgets)
        for group_widgets in tab_widgets.values():
            _set_label_widths(group_widgets)

        self.children = {k: widgets.VBox(v) for k, v in tab_widgets.items()}
        self.tab = self._make_group_container(self.children)
        self.top_section = widgets.VBox(top_widgets)
        self.widget = widgets.VBox([
            self.top_section, self.tab, self.run_button, self._run_output,
        ])
        return self.widget


class ArgparseGUI(ArgparseGUIBase):
    # `inputs` keeps the tall two-panel browser: it is the one argument where you
    # accumulate a list across several directories. Every other path argument is
    # covered by the type dispatch in ArgparseGUIBase.
    custom_widgets = {
        "inputs": ReflectionFileSelector,
    }
