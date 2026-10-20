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
#      (jupyter/abismal_demo_layout.py), without touching their own saved layout.
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

    # Use the environment's own C++ runtime in every process of the session. Its
    # conda-forge packages (numpy, gemmi, DIALS) are built against a newer libstdc++
    # than an older system has; pip's torch finds libstdc++ through the system's
    # library path instead. Whichever loads first serves the whole process, so
    # `import torch` before gemmi failed on S3DF with
    #   /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
    # Preloading the environment's runtime makes it the one, in any import order.
    local runtime="" lib
    for lib in libstdc++.so.6 libgcc_s.so.1; do
        [ -e "$install/env/lib/$lib" ] && runtime="${runtime:+$runtime }$install/env/lib/$lib"
    done
    [ -n "$runtime" ] && export LD_PRELOAD="$runtime${LD_PRELOAD:+ $LD_PRELOAD}"

    mkdir -p "$cache" "$demo"
    export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
    export TRITON_CACHE_DIR="$cache/triton"
    export MPLCONFIGDIR="$cache/matplotlib"
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
    [ -e "$demo/hewl" ] || ln -s "$install/demo/hewl" "$demo/hewl"
    mkdir -p "$demo/runs"

    # Registered every session (it takes a second), so a kernel registered by an
    # earlier version of this script picks up the runtime preload too.
    local kernel_env=()
    [ -n "$runtime" ] && kernel_env=(--env LD_PRELOAD "$runtime")
    "$install/env/bin/python" -m ipykernel install --user --name abismal-s3df \
        --display-name "abismal (S3DF)" "${kernel_env[@]}" >/dev/null 2>&1 \
        || echo "abismal demo: could not register the kernel (the shared Jupyter still has it)" >&2

    echo "abismal demo: environment $install/env, notebook $demo/abismal_s3df_demo.ipynb, caches $cache"
}
_abismal_demo_setup
unset -f _abismal_demo_setup
