# Pre-launch script for the abismal demo in S3DF Jupyter (Open OnDemand).
#
# Paste ONE of these into the Jupyter app's custom-script field:
#
#   source <(curl -fsSL https://raw.githubusercontent.com/rs-station/abismal/s3df-demo/deploy/s3df/demo_setup.sh)
#   source /sdf/.../abismal/demo_setup.sh        # the copy in the shared install; no internet needed
#
# It must be sourced, not run (`curl ... | bash` runs it in a throwaway subshell and the
# environment it sets up is lost before Jupyter starts). It therefore never exits and
# never turns on `set -e`, either of which would end the launcher's own shell.
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
#      other than the shared one.
#
# Override the defaults by setting these before the source line:
#   ABISMAL_S3DF_PREFIX   the shared install (default below)
#   ABISMAL_DEMO_DIR      where the user's demo copy goes (default ~/abismal-demo)
#   ABISMAL_CACHE_DIR     where compile caches go (default: user scratch, else ~/.cache)

_abismal_demo_setup() {
    local prefix="${ABISMAL_S3DF_PREFIX:-/sdf/group/CHANGE_ME/abismal}"
    local demo="${ABISMAL_DEMO_DIR:-$HOME/abismal-demo}"
    local scratch="/sdf/scratch/users/${USER:0:1}/${USER}"
    local cache="${ABISMAL_CACHE_DIR:-}"
    if [ -z "$cache" ]; then
        if [ -d "$scratch" ] && [ -w "$scratch" ]; then cache="$scratch/.cache/abismal"; else cache="$HOME/.cache/abismal"; fi
    fi

    if [ ! -f "$prefix/activate.sh" ]; then
        echo "abismal demo: no shared install at $prefix (set ABISMAL_S3DF_PREFIX); continuing without it" >&2
        return 1
    fi
    # shellcheck source=/dev/null
    source "$prefix/activate.sh"

    mkdir -p "$cache" "$demo"
    export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
    export TRITON_CACHE_DIR="$cache/triton"
    export MPLCONFIGDIR="$cache/matplotlib"
    export ABISMAL_DEMO_DIR="$demo"

    # The user's own copy of the notebook, so edits and outputs stay theirs; never
    # overwrite one they already have.
    [ -e "$demo/abismal_s3df_demo.ipynb" ] || cp "$prefix/abismal_s3df_demo.ipynb" "$demo/"
    [ -e "$demo/hewl" ] || ln -s "$prefix/share/demo/hewl" "$demo/hewl"
    mkdir -p "$demo/runs"

    local kernels="${XDG_DATA_HOME:-$HOME/.local/share}/jupyter/kernels/abismal-s3df"
    if [ ! -f "$kernels/kernel.json" ]; then
        "$prefix/env/bin/python" -m ipykernel install --user --name abismal-s3df \
            --display-name "abismal (S3DF)" >/dev/null 2>&1 \
            || echo "abismal demo: could not register the kernel (the shared Jupyter still has it)" >&2
    fi

    echo "abismal demo: environment $prefix/env, notebook $demo/abismal_s3df_demo.ipynb, caches $cache"
}
_abismal_demo_setup
unset -f _abismal_demo_setup
