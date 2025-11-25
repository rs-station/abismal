from string import Template 
from IPython.display import display,HTML
import reciprocalspaceship as rs
from threading import Thread
import pandas as pd
from os.path import exists
from abismal.command_line.abismal import run_abismal

viewer_template = """<!doctype html>
<html lang="en">
<head>
  <title>UglyMol</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, user-scalable=no">
  <style>
   * { margin: 0; padding: 0; box-sizing: border-box; }
   html, body { 
     width: 100%;
     height: 600px;
     overflow: hidden;
     font-family: sans-serif;
     background-color: black;
   }
   #viewer {
     width: 100%;
     height: 100%;
     position: relative;
   }
   #hud {
     font-size: 15px;
     color: #ddd;
     background-color: rgba(0,0,0,0.6);
     text-align: center;
     position: absolute;
     top: 10px;
     left: 50%;
     transform: translateX(-50%);
     padding: 2px 8px;
     border-radius: 5px;
     z-index: 9;
     white-space: pre-line;
   }
   #hud u { padding: 0 8px; text-decoration: none;
            border: solid; border-width: 1px 0; }
   #hud s { padding: 0 8px; text-decoration: none; opacity: 0.5; }
   #help {
     display: none;
     font-size: 16px;
     color: #eee;
     background-color: rgba(0,0,0,0.7);
     position: absolute;
     left: 20px;
     top: 50%;
     transform: translateY(-50%);
     cursor: default;
     padding: 5px;
     border-radius: 5px;
     z-index: 9;
     white-space: pre-line;
   }
   #inset {
     width: 200px;
     height: 200px;
     background-color: #888;
     position: absolute;
     right: 0;
     bottom: 0;
     z-index: 2;
     display: none;
   }
   a { color: #59C; }
  </style>
</head>
<body>
  <div id="viewer">
    <header id="hud" onmousedown="event.stopPropagation();"
                     ondblclick="event.stopPropagation();">Loading...</header>
    <footer id="help"></footer>
    <div id="inset"></div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/uglymol@0.7.2/uglymol.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mtz@0.1.0/mtz.min.js"></script> 

  <script>
    // Fix for macOS Command key detection
    (function() {
      var originalAddEventListener = EventTarget.prototype.addEventListener;
      EventTarget.prototype.addEventListener = function(type, listener, options) {
        if (type === 'mousedown' || type === 'mousemove' || type === 'mouseup') {
          var wrappedListener = function(e) {
            // Map metaKey (Command on Mac) to ctrlKey for compatibility
            if (e.metaKey && !e.ctrlKey) {
              Object.defineProperty(e, 'ctrlKey', {
                get: function() { return true; }
              });
            }
            return listener.call(this, e);
          };
          originalAddEventListener.call(this, type, wrappedListener, options);
        } else {
          originalAddEventListener.call(this, type, listener, options);
        }
      };
    })();
  </script>

  <script>
    (function initUglyMol() {
      // Check if UM is available
      if (typeof UM === 'undefined') {
        console.log("UM not ready, retrying...");
        setTimeout(initUglyMol, 100);
        return;
      }
      
      console.log("UM is ready!", UM);
      document.getElementById('hud').textContent = "Initializing viewer...";
      
      try {
        var V = new UM.Viewer({viewer: "viewer", hud: "hud", help: "help"});
        console.log("Viewer created:", V);
        
        /*Customizations to Default Settings*/
        V.config.map_radius = 12;
        V.config.water_style = "cross";
        
        document.getElementById('hud').textContent = "Loading PDB...";
        
        V.load_pdb("/files/$pdb_file");
        
        console.log("PDB loading initiated");
        
        // Wait for GemmiMtz to be available
        if (typeof GemmiMtz === 'undefined') {
          console.log("GemmiMtz not ready yet");
          document.getElementById('hud').textContent = "Waiting for MTZ loader...";
          setTimeout(function() {
            loadMtz(V);
          }, 500);
        } else {
          loadMtz(V);
        }
        
      } catch(e) {
        console.error("Error:", e);
        document.getElementById('hud').textContent = "Error: " + e.message;
      }
    })();
    
    function loadMtz(V) {
      try {
        console.log("Loading MTZ...");
        document.getElementById('hud').textContent = "Loading MTZ...";
        
        GemmiMtz().then(function(Module) {
          console.log("GemmiMtz module loaded");
          UM.load_maps_from_mtz(Module, V, "/files/$mtz_file", 
                                $map_keys);
          console.log("MTZ loading initiated");
        });
      } catch(e) {
        console.error("MTZ loading error:", e);
        document.getElementById('hud').textContent = "MTZ Error: " + e.message;
      }
    }
  </script>
</body>
</html>"""


class UglyMolViewer():
    def __init__(self, pdb_file=None, mtz_file=None):
        self.pdb_file = pdb_file
        self.mtz_file = mtz_file

    @property
    def map_keys(self):
        defaults = [
            '2FOFCWT',
            'PH2FOFCWT',
            'ANOM',
            'PANOM',
        ]

        if self.pdb_file is None:
            return None
        ds = rs.read_mtz(self.mtz_file)
        keys = [k for k in defaults if k in ds]
        return keys

    @property
    def template_kwargs(self):
        kwargs = {
            'mtz_file' : self.mtz_file,
            'pdb_file' : self.pdb_file,
            'map_keys' : self.map_keys,
        }
        return kwargs

    @property
    def html(self):
        return Template(viewer_template).substitute(self.template_kwargs)

    def display(self):
        return display(HTML(self.html, metadata={'isolated' : True}))

import argparse
from ipywidgets import widgets

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


class ArgparseGUI:
    def __init__(self, parser=None):
        self.parser = parser
        if parser is None:
            from abismal.command_line.parser import parser as abismal_parser
            self.parser = abismal_parser
        self.polling_period = 5. #seconds


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

    def to_parser(self):
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
        return self.parser.parse_args(map(str, args))

    @staticmethod
    def action_to_widget(action):
        name = ArgparseGUI.action_to_name(action)
        if isinstance(action, argparse._StoreTrueAction):
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
        from abismal.command_line.abismal import run_abismal
        import tf_keras as tfk
        
        # Monkey patch to force keras format
        original_save = tfk.saving.save_model
        
        def patched_save(model, filepath, *args, **kwargs):
            kwargs['save_format'] = 'keras'
            if not filepath.endswith('.keras'):
                filepath = filepath.rsplit('.', 1)[0] + '.keras'
            return original_save(model, filepath, *args, **kwargs)
        
        tfk.saving.save_model = patched_save
        
        try:
            parser = self.to_parser()
            run_abismal(parser)
        finally:
            tfk.saving.save_model = original_save
            parser = self.to_parser()
            run_abismal(parser)

    def to_widget(self):
        self.run_button = widgets.Button(
            description='Run Abismal',
            tooltip='Run Abismal merging',
        )
        self.run_button.on_click(self.run_abismal)
        epochs = 30
        all_widgets = {'Required' : []}
        self._all_args = {}
        for group in self.parser._action_groups:
            group_args = []
            group_widgets = []
            for action in group._group_actions:
                if action.required:
                    group_name = 'Required'
                else:
                    group_name = group.title
                if group_name == 'options':
                    #This group only contains `--help`
                    continue
                if group_name not in all_widgets:
                    all_widgets[group_name] = []
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

