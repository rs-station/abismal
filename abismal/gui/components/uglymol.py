from string import Template 
from IPython.display import display,HTML
import reciprocalspaceship as rs

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


