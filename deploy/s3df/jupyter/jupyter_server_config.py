# Jupyter configuration for the abismal S3DF demo. demo_setup.sh puts this directory on
# JUPYTER_CONFIG_PATH, so it applies only to sessions started with the demo; it reads
# nothing and changes nothing unless ABISMAL_DEMO_DIR is set.
#
# It makes JupyterLab open on the demo notebook, in ~/abismal-demo, and not on whatever
# the user had open in earlier sessions:
#   - JupyterLab's saved layouts ("workspaces") are kept in the demo directory, so a demo
#     session neither restores nor overwrites the user's own;
#   - the default layout is rewritten at every launch with the demo notebook open and the
#     file browser in the demo directory;
#   - the server's start page and file browser start there too, for a session that is
#     opened at the server's root rather than at /lab.
# Paths in a layout are relative to the server's root directory, which the app chooses on
# its command line, so it is read from there (or is the launch directory).
import json
import os
import sys

c = get_config()  # noqa: F821 (provided by Jupyter when it loads this file)


def _root_dir():
    names = ("--ServerApp.root_dir", "--notebook-dir", "--NotebookApp.notebook_dir",
             "--ServerApp.notebook_dir")
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        for name in names:
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
            if arg == name and i + 1 < len(args):
                return args[i + 1]
    return os.getcwd()


def _configure(demo):
    root = os.path.realpath(os.path.expanduser(_root_dir()))
    demo = os.path.realpath(os.path.expanduser(demo))
    notebook = os.path.join(demo, "abismal_s3df_demo.ipynb")
    if not os.path.exists(notebook):
        return
    rel_dir = os.path.relpath(demo, root)
    rel_notebook = os.path.relpath(notebook, root)
    if rel_dir.startswith(".."):
        return  # the demo is outside what this server can show
    rel_dir = "" if rel_dir == "." else rel_dir

    workspaces = os.path.join(demo, ".jupyter-workspaces")
    os.makedirs(workspaces, exist_ok=True)
    c.LabApp.workspaces_dir = workspaces
    c.FileContentsManager.preferred_dir = rel_dir
    c.ServerApp.default_url = "/lab/tree/" + rel_notebook

    widget = "notebook:" + rel_notebook
    layout = {
        "data": {
            "layout-restorer:data": {
                "main": {"dock": {"type": "tab-area", "currentIndex": 0, "widgets": [widget]},
                         "current": widget},
                "down": {"size": 0, "collapsed": True, "widgets": []},
                "left": {"collapsed": False, "visible": True, "current": "filebrowser",
                         "widgets": ["filebrowser", "running-sessions", "@jupyterlab/toc:plugin",
                                     "extensionmanager.main-view"]},
                "right": {"collapsed": True, "visible": True,
                          "widgets": ["jp-property-inspector", "debugger-sidebar"]},
            },
            "file-browser-filebrowser:cwd": {"path": rel_dir},
            widget: {"data": {"path": rel_notebook, "factory": "Notebook"}},
        },
        "metadata": {"id": "default"},
    }
    from jupyterlab_server.workspaces_handler import WorkspacesManager
    WorkspacesManager(workspaces).save("default", json.dumps(layout))


_demo = os.environ.get("ABISMAL_DEMO_DIR")
if _demo:
    try:
        _configure(_demo)
    except Exception as error:  # never stop Jupyter from starting over the demo layout
        print(f"abismal demo: could not set up the JupyterLab layout: {error!r}", file=sys.stderr)
