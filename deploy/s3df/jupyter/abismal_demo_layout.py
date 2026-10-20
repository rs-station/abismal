"""
A Jupyter server extension for the abismal S3DF demo: open JupyterLab on the demo notebook.

demo_setup.sh puts this directory on JUPYTER_CONFIG_PATH, where
jupyter_server_config.d/abismal_demo.json enables the extension, and on PYTHONPATH, so it
can be imported. Both are found even when the app starts the server with its own
--config file, which replaces every other config file. It runs once the server is set
up, when the server's root directory is known however the app chose it (on the command
line, in its own config file, or as the launch directory), and:
  - writes JupyterLab's default layout with the demo notebook open and the file browser
    in the demo directory, into the demo's own layout store;
  - points the server's start page and file browser at the demo, for a session opened at
    the server's root rather than at /lab.
"""
import json
import os


def _jupyter_server_extension_points():
    return [{"module": "abismal_demo_layout"}]


def _load_jupyter_server_extension(serverapp):
    log = serverapp.log
    try:
        _apply(serverapp, log)
    except Exception as error:  # never stop Jupyter from starting over the demo layout
        log.warning(f"abismal demo: could not set up the JupyterLab layout: {error!r}")


def _apply(serverapp, log):
    demo = os.environ.get("ABISMAL_DEMO_DIR")
    workspaces = os.environ.get("JUPYTERLAB_WORKSPACES_DIR")
    if not demo or not workspaces:
        return
    root = os.path.realpath(os.path.expanduser(serverapp.root_dir))
    demo = os.path.realpath(os.path.expanduser(demo))
    notebook = os.path.join(demo, "abismal_s3df_demo.ipynb")
    if not os.path.exists(notebook):
        log.warning(f"abismal demo: no notebook at {notebook}; JupyterLab opens as usual")
        return
    rel_dir = os.path.relpath(demo, root)
    if rel_dir.startswith(".."):
        log.warning(f"abismal demo: {demo} is outside the server's root {root}; JupyterLab opens as usual")
        return
    rel_dir = "" if rel_dir == "." else rel_dir
    rel_notebook = os.path.relpath(notebook, root)

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

    serverapp.contents_manager.preferred_dir = rel_dir
    default_url = serverapp.base_url.rstrip("/") + "/lab/tree/" + rel_notebook
    serverapp.default_url = default_url
    if getattr(serverapp, "web_app", None) is not None:
        serverapp.web_app.settings["default_url"] = default_url
    log.info(f"abismal demo: JupyterLab opens on {notebook} (server root {root}, layouts in {workspaces})")
