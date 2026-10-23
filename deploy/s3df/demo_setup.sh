# Pre-launch script for the abismal demo in S3DF Jupyter (Open OnDemand).
#
# Paste this into the Jupyter app's custom-script field, with the path of the shared
# clone that deploy/s3df/install_env.sh was run in:
#
#   source /sdf/group/<facility>/abismal/deploy/s3df/demo_setup.sh
#
# It must be sourced from that clone, not run and not piped from elsewhere: it finds
# the environment next to itself, and the environment it sets up must outlive it to
# reach Jupyter. It therefore never exits and never turns on `set -e`, either of which
# would end the launcher's own shell.
#
# What it does, for the user who launches the session (nothing outside their own space
# is written):
#   1. everything setup.sh does: puts the shared abismal environment first on PATH (so
#      JupyterLab, its widget support and the kernel all come from it), preloads its C++
#      runtime, sends compile caches to the user's scratch, and registers an
#      "abismal (S3DF)" Jupyter kernel;
#   2. creates ~/abismal-demo with a copy of the demo notebook, and links the example
#      data there (read-only, shared);
#   3. makes JupyterLab open on the demo notebook instead of the user's last session
#      (jupyter/abismal_demo_layout.py), without touching their own saved layout.
#
# To use abismal without the demo, source setup.sh instead.
#
# Override the defaults by setting these before the source line:
#   ABISMAL_DEMO_DIR      where the user's demo copy goes (default ~/abismal-demo)
#   ABISMAL_CACHE_DIR     where compile caches go (default: user scratch, else ~/.cache)

_abismal_demo_setup() {
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    local install="$here/install"
    local demo="${ABISMAL_DEMO_DIR:-$HOME/abismal-demo}"

    # The environment itself: PATH, the C++ runtime, caches and the kernel.
    source "$here/setup.sh" || { echo "abismal demo: continuing without the environment" >&2; return 1; }

    mkdir -p "$demo"
    export ABISMAL_DEMO_DIR="$demo"
    # Open JupyterLab on the demo notebook rather than the user's last session: keep the
    # demo's saved layouts apart from the user's own, and let the abismal_demo_layout
    # server extension (jupyter/) write the demo layout. Their own settings still apply.
    export JUPYTERLAB_WORKSPACES_DIR="$demo/.jupyter-workspaces"
    mkdir -p "$JUPYTERLAB_WORKSPACES_DIR"
    export JUPYTER_CONFIG_PATH="$here/jupyter${JUPYTER_CONFIG_PATH:+:$JUPYTER_CONFIG_PATH}"
    export PYTHONPATH="$here/jupyter${PYTHONPATH:+:$PYTHONPATH}"

    # The user's own copy of the notebook, so edits and outputs stay theirs; never
    # overwrite one they already have.
    [ -e "$demo/abismal_s3df_demo.ipynb" ] || cp "$here/abismal_s3df_demo.ipynb" "$demo/"
    [ -e "$demo/hewl" ] || [ -L "$demo/hewl" ] || ln -s "$install/demo/hewl" "$demo/hewl"
    mkdir -p "$demo/runs"

    echo "abismal demo: notebook $demo/abismal_s3df_demo.ipynb"
}
_abismal_demo_setup
_abismal_status=$?
unset -f _abismal_demo_setup
eval "unset _abismal_status; (exit $_abismal_status)"
