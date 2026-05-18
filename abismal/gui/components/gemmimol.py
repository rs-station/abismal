import uuid
from string import Template
from IPython.display import display,HTML
import reciprocalspaceship as rs

viewer_template = """<!doctype html>
<html lang="en">
<head>
  <title>GemmiMol</title>
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

  <script src="https://cdn.jsdelivr.net/npm/gemmimol@0.8.8/gemmimol.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/gemmimol@0.8.8/vendor/wasm/gemmi.js"></script>

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
    var V = null;
    window.ABISMAL_VIEWER_ID = '$viewer_id';

    // Receive reload commands from the parent notebook so the camera is preserved.
    window.addEventListener('message', function(event) {
      if (!event.data || event.data.type !== 'reload' || !V) return;
      var msg = event.data;
      // Dispose existing models and maps before loading the new epoch.
      while (V.model_bags.length > 0) {
        V.clear_model_objects(V.model_bags[0]);
        V.model_bags.splice(0, 1);
      }
      while (V.map_bags.length > 0) {
        V.clear_el_objects(V.map_bags[0]);
        V.map_bags.splice(0, 1);
      }
      document.getElementById('hud').textContent = 'Loading PDB...';
      V.load_model(msg.pdb_file, {stay: true});
      loadMtz(msg.mtz_file, msg.map_keys);
    });

    function loadMtz(mtzPath, mapKeys) {
      document.getElementById('hud').textContent = 'Loading MTZ...';
      Gemmi().then(function(Module) {
        GM.load_maps_from_mtz(Module, V, mtzPath, mapKeys);
      });
    }

    (function initGemmiMol() {
      if (typeof GM === 'undefined') {
        setTimeout(initGemmiMol, 100);
        return;
      }
      try {
        V = new GM.Viewer({viewer: "viewer", hud: "hud", help: "help"});
        V.config.map_radius = 12;
        V.config.water_style = "cross";
        document.getElementById('hud').textContent = 'Loading PDB...';
        V.load_model('$pdb_file');
        if (typeof Gemmi === 'undefined') {
          setTimeout(function() { loadMtz('$mtz_file', $map_keys); }, 500);
        } else {
          loadMtz('$mtz_file', $map_keys);
        }
      } catch(e) {
        document.getElementById('hud').textContent = 'Error: ' + e.message;
      }
    })();
  </script>
</body>
</html>"""


class GemmiMolViewer():
    def __init__(self, pdb_file=None, mtz_file=None, viewer_id=None,
                 pdb_url=None, mtz_url=None):
        # pdb_file/mtz_file are paths the kernel will open (rs.read_mtz);
        # pdb_url/mtz_url are paths the browser will request under /files/.
        # On JupyterLab where server root_dir != kernel cwd, these differ.
        self.pdb_file = pdb_file
        self.mtz_file = mtz_file
        self.pdb_url = pdb_url if pdb_url is not None else pdb_file
        self.mtz_url = mtz_url if mtz_url is not None else mtz_file
        self.viewer_id = viewer_id or str(uuid.uuid4())

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
        return {
            'mtz_file' : self.mtz_url,
            'pdb_file' : self.pdb_url,
            'map_keys' : self.map_keys,
            'viewer_id': self.viewer_id,
        }

    @property
    def html(self):
        return Template(viewer_template).substitute(self.template_kwargs)

    def display(self):
        return display(HTML(self.html, metadata={'isolated' : True}))

