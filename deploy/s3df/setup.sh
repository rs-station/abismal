# Put the shared abismal environment on your path, for running abismal your own way: the CLI
# in a terminal or a batch job, or the GUI notebook in Jupyter.
#
# Source it from the shared clone that deploy/s3df/install_env.sh was run in, e.g. in a
# terminal, at the top of a batch script, in ~/.bashrc, or in the Jupyter app's custom-script
# field:
#
#   source /sdf/group/<facility>/abismal/deploy/s3df/setup.sh
#
# It must be sourced, not run: the environment it sets up has to outlive it. It therefore
# never exits and never turns on `set -e`, either of which would end the calling shell.
#
# What it does, writing nothing outside your own space:
#   1. puts the shared environment first on PATH: `abismal`, `abismal.gui`, `abismal.cchalf`,
#      python, jupyter and the rest come from it;
#   2. preloads the environment's own C++ runtime (see below);
#   3. sends the torch.compile, triton and matplotlib caches to your scratch directory rather
#      than your small home quota;
#   4. registers an "abismal (S3DF)" Jupyter kernel, so a Jupyter you start some other way can
#      run the GUI notebook too. Set ABISMAL_REGISTER_KERNEL=0 to skip it.
#
# Unlike demo_setup.sh it creates no demo directory and leaves Jupyter's layout alone.
#
# Override the defaults by setting these before the source line:
#   ABISMAL_CACHE_DIR        where compile caches go (default: your scratch, else ~/.cache)
#   ABISMAL_REGISTER_KERNEL  0 to skip registering the Jupyter kernel (default 1)

_abismal_setup() {
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    local install="$here/install"
    local scratch="/sdf/scratch/users/${USER:0:1}/${USER}"
    local cache="${ABISMAL_CACHE_DIR:-}"
    if [ -z "$cache" ]; then
        if [ -d "$scratch" ] && [ -w "$scratch" ]; then cache="$scratch/.cache/abismal"; else cache="$HOME/.cache/abismal"; fi
    fi

    if [ ! -x "$install/env/bin/python" ]; then
        echo "abismal: no environment at $install. Source this file from the clone" \
             "that deploy/s3df/install_env.sh was run in; continuing without it" >&2
        return 1
    fi
    export PATH="$install/env/bin:$PATH"

    # Use the environment's own C++ runtime in every process. Its conda-forge packages
    # (numpy, gemmi, DIALS) are built against a newer libstdc++ than an older system has;
    # pip's torch finds libstdc++ through the system's library path instead. Whichever loads
    # first serves the whole process, so `import torch` before gemmi failed on S3DF with
    #   /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
    # Preloading the environment's runtime makes it the one, in any import order.
    local runtime="" lib
    for lib in libstdc++.so.6 libgcc_s.so.1; do
        [ -e "$install/env/lib/$lib" ] && runtime="${runtime:+$runtime }$install/env/lib/$lib"
    done
    [ -n "$runtime" ] && export LD_PRELOAD="$runtime${LD_PRELOAD:+ $LD_PRELOAD}"

    mkdir -p "$cache"
    export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
    export TRITON_CACHE_DIR="$cache/triton"
    export MPLCONFIGDIR="$cache/matplotlib"
    export ABISMAL_SETUP_DONE="$install"

    # Registered every time (it takes a second), so a kernel registered by an earlier
    # version of this script picks up the runtime preload too.
    if [ "${ABISMAL_REGISTER_KERNEL:-1}" != 0 ]; then
        local kernel_env=()
        [ -n "$runtime" ] && kernel_env=(--env LD_PRELOAD "$runtime")
        "$install/env/bin/python" -m ipykernel install --user --name abismal-s3df \
            --display-name "abismal (S3DF)" "${kernel_env[@]}" >/dev/null 2>&1 \
            || echo "abismal: could not register the Jupyter kernel" >&2
    fi

    echo "abismal: environment $install/env on PATH, caches in $cache"
}
_abismal_setup
# Hand the status to `source` (so demo_setup.sh can stop when there is no environment)
# without exiting the caller's shell.
_abismal_status=$?
unset -f _abismal_setup
eval "unset _abismal_status; (exit $_abismal_status)"
