from abismal.command_line.parser import parser as abismal_parser
import argparse
from ipywidgets import widgets
from abismal.gui.components.file_selector import (
    ReflectionFileSelector,
    PhenixFileSelector,
    ColabReflectionFileSelector,
    ColabPhenixFileSelector,
    _is_colab,
)


def _label(text):
    return widgets.HTML(
        value=f'<div style="text-align:right;line-height:32px;padding-right:6px">{text}</div>',
        layout=widgets.Layout(width='120px', min_width='120px'),
    )


def _set_label_widths(widget_list):
    """Set label column width for a group of widgets based on the longest label text."""
    labelable = [w for w in widget_list if hasattr(w, '_label_text')]
    if not labelable:
        return
    max_chars = max(len(w._label_text) for w in labelable)
    width = f'{max_chars * 8 + 24}px'
    for w in labelable:
        lbl = w.children[0]
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
    skipped_actions = [
        "help",
        "list_devices",
        "run_eagerly",
        "debug",
        "embed",
        "keras_verbosity",
    ]

    def __init__(self, parser=None):
        self.parser = parser if parser is not None else abismal_parser

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

    def run_abismal(self, button=None):
        from abismal.gui.runner import AbismalRunner
        from abismal.gui.cleanup import find_abismal_outputs
        parsed = self.to_parser()
        out_dir = str(parsed.out_dir)
        has_phenix = getattr(parsed, "eff_files", None) is not None

        runner = AbismalRunner.attach(out_dir, has_phenix=has_phenix)
        if runner is not None:
            runner._append_log(
                f'[Reconnected to running process PID {runner._pid}]\n'
            )
            runner.resume()
            self.widget.children = (
                self.top_section, self.tab, self.run_button,
                self._run_output, runner.to_widget(),
            )
            return

        existing = find_abismal_outputs(out_dir)
        if existing:
            self._show_overwrite_confirm(out_dir, existing, parsed, has_phenix)
            return

        self._launch_runner(parsed, has_phenix)

    def _launch_runner(self, parsed, has_phenix):
        from abismal.gui.runner import AbismalRunner
        out_dir = str(parsed.out_dir)
        args = self.to_args()
        runner = AbismalRunner(
            args, out_dir, has_phenix=has_phenix,
            total_epochs=parsed.epochs,
        )
        runner.start()
        self.widget.children = (
            self.top_section, self.tab, self.run_button,
            self._run_output, runner.to_widget(),
        )

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
            # Sentinel: prove click delivery independent of widget rendering.
            # On Colab, `tail -f /tmp/abismal_click.log` from a terminal will
            # show whether the kernel is receiving on_click events at all.
            import datetime, traceback
            with open('/tmp/abismal_click.log', 'a') as _f:
                _f.write(f'{datetime.datetime.now().isoformat()} click\n')
            self._run_output.clear_output()
            with self._run_output:
                try:
                    self.run_abismal(button)
                except Exception:
                    traceback.print_exc()
                    with open('/tmp/abismal_click.log', 'a') as _f:
                        _f.write(traceback.format_exc())
        self.run_button.on_click(_on_run_click)
        top_widgets = []
        out_dir_widget = None
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
                if action.dest == 'out_dir':
                    out_dir_widget = widget
                elif action.required:
                    top_widgets.append(widget)
                else:
                    group_name = group.title
                    if group_name not in tab_widgets:
                        tab_widgets[group_name] = []
                    tab_widgets[group_name].append(widget)

        if out_dir_widget is not None:
            top_widgets.append(out_dir_widget)

        _set_label_widths(top_widgets)
        for group_widgets in tab_widgets.values():
            _set_label_widths(group_widgets)

        self.children = {k: widgets.VBox(v) for k, v in tab_widgets.items()}
        self.tab = widgets.Tab(children=list(self.children.values()))
        # set_title works in both ipywidgets 7 (Colab default) and 8; the
        # `titles=` kwarg is ipywidgets 8 only and is silently dropped on 7.
        for i, title in enumerate(self.children.keys()):
            self.tab.set_title(i, title)
        self.top_section = widgets.VBox(top_widgets)
        self.widget = widgets.VBox([
            self.top_section, self.tab, self.run_button, self._run_output,
        ])
        return self.widget


class JupyterArgparseGUI(ArgparseGUIBase):
    custom_widgets = {
        "inputs": ReflectionFileSelector,
        "eff_files": PhenixFileSelector,
    }


class ColabArgparseGUI(ArgparseGUIBase):
    custom_widgets = {
        "inputs": ColabReflectionFileSelector,
        "eff_files": ColabPhenixFileSelector,
    }


ArgparseGUI = ColabArgparseGUI if _is_colab() else JupyterArgparseGUI
