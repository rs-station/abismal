from abismal.command_line.parser import parser as abismal_parser
import argparse
from ipywidgets import widgets
from time import sleep
from abismal.gui.components.file_selector import ReflectionFileSelector,PhenixFileSelector

class Text(widgets.Box):
    def __init__(self, **kwargs):
        description = ''
        if 'description' in kwargs:
            description = kwargs.pop('description')
        children = [
            widgets.Label(description),
            widgets.Text(**kwargs),
        ]
        super().__init__(children)

    @property
    def label(self):
        return self.children[0]

    @property
    def text(self):
        return self.children[1]

    @property
    def value(self):
        return self.text.value

class Dropdown(widgets.Box):
    def __init__(self, **kwargs):
        description = ''
        if 'description' in kwargs:
            description = kwargs.pop('description')
        children = [
            widgets.Label(description),
            widgets.Dropdown(**kwargs),
        ]
        super().__init__(children)

    @property
    def label(self):
        return self.children[0]

    @property
    def dropdown(self):
        return self.children[1]

    @property
    def value(self):
        return self.dropdown.value

class ArgparseGUIBase:
    custom_actions = {}
    skipped_actions = []
    def __init__(self, parser=None):
        self.parser = parser
        if parser is None:
            self.parser = abismal_parser
        self.polling_period = 5. #seconds

    def poll(self):
        if self._clicked:
            self.run_abismal()
        self._clicked = False

    def polling_loop(self):
        while True:
            self.poll();
            sleep(self.polling_period)

    @staticmethod
    def is_required(*args, **kwargs):
        if args[0][0] != '-':
            return True
        elif 'required' in kwargs and kwargs['required']:
            return True
        return False

    @staticmethod
    def action_to_name(action):
        if action.metavar is not None:
            return action.metavar
        return action.dest

    def to_args(self):
        args = []
        for k,v in self._all_args.items():
            v = v.value
            if v == '':
                continue
            if isinstance(k, argparse._StoreTrueAction):
                if v == False:
                    continue
                else: 
                    args.append(k.option_strings[0])
            else:
                if len(k.option_strings) > 0:
                    args.append(k.option_strings[0])
                args.append(v)
        return list(map(str, args))

    def to_parser(self):
        return self.parser.parse_args(self.to_args())

    def action_to_widget(self, action, name=None):
        if name is None:
            name = ArgparseGUI.action_to_name(action)
        if name in self.custom_widgets:
            return self.custom_widgets[name](action, name=name)
        elif isinstance(action, argparse._StoreTrueAction):
            return widgets.ToggleButton(
                    value=False,
                    description=name,
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip=action.help,
                )
        elif isinstance(action, argparse._StoreTrueAction):
            return widgets.ToggleButton(
                    value=False,
                    description=name,
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip=action.help,
                )
        elif isinstance(action, argparse._StoreAction):
            if action.choices is not None:
                return Dropdown(
                    options=action.choices,
                    value=action.type(action.default),
                    description=name,
                    tooltip=action.help,
                    disabled=False,
                )
            else:
                # Fallback text field
                return Text(
                    placeholder=str(action.default),
                    tooltip=action.help,
                    description=name,
                )


    def run_abismal(self, *args, **kwargs):
        from subprocess import call
        args = self.to_args()
        self.current_process = call(['abismal'] + args)

    def _set_clicked(self, *args, **kwargs):
        self._clicked = True

    def to_widget(self):
        self._clicked = False
        self.run_button = widgets.Button(
            description='Run Abismal',
            tooltip='Run Abismal merging',
        )
        self.run_button.on_click(self.run_abismal)
        all_widgets = {'Required' : []}
        self._all_args = {}
        for group in self.parser._action_groups:
            group_args = []
            group_widgets = []
            for action in group._group_actions:
                name = self.action_to_name(action)
                if name in self.skipped_actions:
                    continue
                if action.required:
                    group_name = 'Required'
                else:
                    group_name = group.title
                if group_name not in all_widgets:
                    all_widgets[group_name] = []
                if name in self.custom_actions:
                    widget = self.custom_actions[name](action)
                else:
                    widget = self.action_to_widget(action)

                self._all_args[action] = widget
                all_widgets[group_name].append(widget)

        self.children = {k:widgets.VBox(v) for k,v in all_widgets.items()}
        self.tab = widgets.Tab(
            children = list(self.children.values()),
            titles = list(self.children.keys()),
        )
        self.widget = widgets.VBox([
            self.tab,
            self.run_button,
        ])
        return self.widget

    @staticmethod
    def action_to_file_selector(action, name=None):
        from solara.components import FileBrowswer
        if name is None:
            name = ArgparseGUI.action_to_name(action)
        fb = FileBrowswer('.')
        return fb.widgets

class ArgparseGUI(ArgparseGUIBase):
    custom_widgets={
        'inputs' : ReflectionFileSelector,
        'eff_files' : PhenixFileSelector,
    }
    skipped_actions = [
        'help',
        'list_devices',
        'jit_compile',
        'run_eagerly',
        'debug',
    ]

