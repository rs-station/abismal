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
#   1. puts the shared abismal environment first on PATH, so JupyterLab, its widget
#      support and the kernel all come from it;
#   2. sends the torch.compile, triton and matplotlib caches to the user's scratch, not
#      their small home quota;
#   3. creates ~/abismal-demo with a copy of the demo notebook, and links the example
#      data there (read-only, shared);
#   4. registers an "abismal (S3DF)" Jupyter kernel, in case the session starts a Jupyter
#      other than the shared one;
#   5. makes JupyterLab open on the demo notebook instead of the user's last session
#      (jupyter/jupyter_server_config.py), without touching their own saved layout.
#
# Override the defaults by setting these before the source line:
#   ABISMAL_DEMO_DIR      where the user's demo copy goes (default ~/abismal-demo)
#   ABISMAL_CACHE_DIR     where compile caches go (default: user scratch, else ~/.cache)

_abismal_demo_setup() {
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    local install="$here/install"
    local demo="${ABISMAL_DEMO_DIR:-$HOME/abismal-demo}"
    local scratch="/sdf/scratch/users/${USER:0:1}/${USER}"
    local cache="${ABISMAL_CACHE_DIR:-}"
    if [ -z "$cache" ]; then
        if [ -d "$scratch" ] && [ -w "$scratch" ]; then cache="$scratch/.cache/abismal"; else cache="$HOME/.cache/abismal"; fi
    fi

    if [ ! -x "$install/env/bin/python" ]; then
        echo "abismal demo: no environment at $install. Source this file from the clone" \
             "that deploy/s3df/install_env.sh was run in; continuing without it" >&2
        return 1
    fi
    export PATH="$install/env/bin:$PATH"

    mkdir -p "$cache" "$demo"
    export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
    export TRITON_CACHE_DIR="$cache/triton"
    export MPLCONFIGDIR="$cache/matplotlib"
    export ABISMAL_DEMO_DIR="$demo"
    # Open JupyterLab on the demo notebook rather than the user's last session; see
    # jupyter/jupyter_server_config.py. Their own Jupyter settings still apply.
    export JUPYTER_CONFIG_PATH="$here/jupyter${JUPYTER_CONFIG_PATH:+:$JUPYTER_CONFIG_PATH}"

    # The user's own copy of the notebook, so edits and outputs stay theirs; never
    # overwrite one they already have.
    [ -e "$demo/abismal_s3df_demo.ipynb" ] || cp "$here/abismal_s3df_demo.ipynb" "$demo/"
    [ -e "$demo/hewl" ] || ln -s "$install/demo/hewl" "$demo/hewl"
    mkdir -p "$demo/runs"

    local kernels="${XDG_DATA_HOME:-$HOME/.local/share}/jupyter/kernels/abismal-s3df"
    if [ ! -f "$kernels/kernel.json" ]; then
        "$install/env/bin/python" -m ipykernel install --user --name abismal-s3df \
            --display-name "abismal (S3DF)" >/dev/null 2>&1 \
            || echo "abismal demo: could not register the kernel (the shared Jupyter still has it)" >&2
    fi

    echo "abismal demo: environment $install/env, notebook $demo/abismal_s3df_demo.ipynb, caches $cache"
}
_abismal_demo_setup
unset -f _abismal_demo_setup
