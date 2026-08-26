import base64
import uuid
from pathlib import Path
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

    // The pdb and mtz arrive as base64 rather than as URLs. gemmimol fetches
    // whatever path it is given, and the only paths a notebook can offer are
    // /files/ ones, which the jupyter server serves solely from under its
    // root_dir -- so an out_dir anywhere else had no URL at all, and Colab has no
    // /files/ endpoint whatsoever. A blob URL is same-origin with this frame and
    // needs no server, so load_model and load_maps_from_mtz work unchanged.
    var BLOB_URLS = [];
    function blobUrl(b64) {
      var binary = atob(b64);
      var bytes = new Uint8Array(binary.length);
      for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      var url = URL.createObjectURL(new Blob([bytes]));
      BLOB_URLS.push(url);
      return url;
    }

    function releaseBlobUrls() {
      // Each epoch mints two more, and a long run would otherwise hold every
      // epoch's files in memory for the life of the tab.
      BLOB_URLS.forEach(function(url) { URL.revokeObjectURL(url); });
      BLOB_URLS = [];
    }

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
      releaseBlobUrls();
      V.load_model(blobUrl(msg.pdb_b64), {stay: true});
      loadMtz(blobUrl(msg.mtz_b64), msg.map_keys);
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
        V.load_model(blobUrl('$pdb_b64'));
        var mtzUrl = blobUrl('$mtz_b64');
        if (typeof Gemmi === 'undefined') {
          setTimeout(function() { loadMtz(mtzUrl, $map_keys); }, 500);
        } else {
          loadMtz(mtzUrl, $map_keys);
        }
      } catch(e) {
        document.getElementById('hud').textContent = 'Error: ' + e.message;
      }
    })();
  </script>
</body>
</html>"""


class GemmiMolViewer():
    """A standalone 3D viewer document with the model and maps embedded in it.

    The files travel as base64 rather than as URLs the browser fetches. The only
    URLs a notebook can offer are /files/ ones, which the jupyter server serves
    solely from under its root_dir -- so results written anywhere else had no URL
    at all, and Colab, which has no /files/ endpoint, never worked. Embedding
    removes the server from the picture entirely; the page is self-contained.

    The cost is size: roughly 4/3 of the two files, so about 1.5 MB for a typical
    torchref epoch and up to ~5 MB for one of phenix's larger mtzs.
    """

    def __init__(self, pdb_file=None, mtz_file=None, viewer_id=None):
        self.pdb_file = pdb_file
        self.mtz_file = mtz_file
        self.viewer_id = viewer_id or str(uuid.uuid4())

    @staticmethod
    def encode(path):
        """A file as base64, safe to drop straight into a javascript string."""
        if path is None:
            return ''
        return base64.b64encode(Path(path).read_bytes()).decode('ascii')

    @property
    def pdb_b64(self):
        return self.encode(self.pdb_file)

    @property
    def mtz_b64(self):
        return self.encode(self.mtz_file)

    # (amplitude, phase) column pairs, in the order the viewer should stack them.
    # gemmimol's load_maps_from_mtz takes one flat list and reads it pairwise, so the
    # pairing has to be preserved exactly.
    #
    # phenix and torchref spell the same maps differently: phenix writes 2FOFCWT/
    # PH2FOFCWT, torchref writes FWT/PHWT. Only phenix's names were listed here, so
    # every --torchref-pdb run loaded the anomalous difference map alone and no 2Fo-Fc
    # map at all -- silently, since a missing column is simply filtered out.
    #
    # The Fo-Fc maps (FOFCWT/PHFOFCWT, DELFWT/PHDELWT) are deliberately not included;
    # this viewer has always shown 2Fo-Fc plus anomalous, and adding a third map is a
    # display change rather than a fix.
    map_columns = (
        ('2FOFCWT', 'PH2FOFCWT'),   # phenix   2Fo-Fc
        ('FWT', 'PHWT'),            # torchref 2Fo-Fc
        ('ANOM', 'PANOM'),          # anomalous difference, spelled the same by both
    )

    @property
    def map_keys(self):
        # The guard is on the mtz, which is what gets opened. It read `pdb_file` until
        # 2026-08-25, so a viewer with a model but no mtz raised instead of returning
        # None.
        if self.mtz_file is None:
            return None
        ds = rs.read_mtz(self.mtz_file)
        keys = []
        for amplitude, phase in self.map_columns:
            # Both halves or neither -- emitting a lone amplitude would shift every
            # later pair by one and mis-assign the phases.
            if amplitude in ds and phase in ds:
                keys += [amplitude, phase]
        return keys

    @property
    def template_kwargs(self):
        return {
            'mtz_b64' : self.mtz_b64,
            'pdb_b64' : self.pdb_b64,
            'map_keys' : self.map_keys,
            'viewer_id': self.viewer_id,
        }

    @property
    def reload_payload(self):
        """What the parent postMessages to update an already-embedded viewer.

        Re-embedding instead would rebuild the iframe and reset the camera.
        """
        return {
            'type': 'reload',
            'pdb_b64': self.pdb_b64,
            'mtz_b64': self.mtz_b64,
            'map_keys': self.map_keys,
        }

    @property
    def html(self):
        return Template(viewer_template).substitute(self.template_kwargs)

    def display(self):
        return display(HTML(self.html, metadata={'isolated' : True}))

