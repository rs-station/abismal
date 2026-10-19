#!/usr/bin/env bash
# Build the abismal environment for the S3DF Jupyter demo inside this clone.
#
# Clone the repository where users can read it (shared, fast storage), then run this
# once as the clone's owner; users only ever read from it:
#
#   git clone -b s3df-demo https://github.com/rs-station/abismal /sdf/group/<facility>/abismal
#   /sdf/group/<facility>/abismal/deploy/s3df/install_env.sh
#
# Everything lands in deploy/s3df/install/ (git-ignored), next to this script:
#
#   install/bin/micromamba   the package manager used to build the environment
#   install/env/             python, torch (CUDA), abismal[gui,torchref] from this clone, jupyterlab
#   install/demo/hewl/       the lysozyme example data and its reference files
#
# Users then source deploy/s3df/demo_setup.sh from this clone; see README.md. To update,
# `git pull` and run this again.
#
# Usage:
#   deploy/s3df/install_env.sh [--torch-index URL] [--with-dials]
#
#   --torch-index  take torch from this PyTorch wheel index, e.g.
#                  https://download.pytorch.org/whl/cu128. Only needed when the GPU
#                  driver is too old for the CUDA build PyPI ships (CUDA 13 needs driver
#                  580 or newer); the script says so at the end if the GPU is not usable.
#   --with-dials   also install DIALS/dxtbx from conda-forge, needed for .expt/.refl
#                  inputs (several GB; .mtz and CrystFEL .stream inputs never need it)
#
# Needs internet access (GitHub, conda-forge, PyPI): run it on a login or interactive node
# that has it, ideally a GPU node so the closing check can see the GPU.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

TORCH_INDEX=""
WITH_DIALS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        --with-dials) WITH_DIALS=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument $1" >&2; usage 1 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
INSTALL="$HERE/install"
ENV="$INSTALL/env"
PY="$ENV/bin/python"
mkdir -p "$INSTALL"
step() { printf '\n== %s\n' "$*"; }

step "micromamba"
if [ ! -x "$INSTALL/bin/micromamba" ]; then
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$INSTALL" bin/micromamba
fi
MAMBA="$INSTALL/bin/micromamba"
export MAMBA_ROOT_PREFIX="$INSTALL/mamba"

step "python environment at $ENV"
if [ ! -x "$PY" ]; then
    "$MAMBA" create -y -q -p "$ENV" -c conda-forge python=3.12 pip
fi
if [ "$WITH_DIALS" = 1 ]; then
    step "dials (conda-forge)"
    "$MAMBA" install -y -q -p "$ENV" -c conda-forge dials
fi

step "abismal[gui,torchref] from $REPO, torch, jupyterlab"
# One pass, so pip resolves torch together with torchref's cap on it. A wheel from an
# extra index carries a local version (+cu128), which outranks PyPI's for the same release.
EXTRA=()
[ -n "$TORCH_INDEX" ] && EXTRA=(--extra-index-url "$TORCH_INDEX")
"$PY" -m pip install -q --no-cache-dir --upgrade "${EXTRA[@]}" "$REPO[gui,torchref]" jupyterlab ipykernel

step "example data"
DEMO="$INSTALL/demo/hewl"
mkdir -p "$DEMO"
"$PY" - "$DEMO" <<'PYEOF'
import json, os, sys, urllib.request
from abismal.gui._launch import EXAMPLE_URL, REFERENCE_API
demo = sys.argv[1]
target = os.path.join(demo, "unmerged.mtz")
if not os.path.exists(target):
    urllib.request.urlretrieve(EXAMPLE_URL, target)
with urllib.request.urlopen(REFERENCE_API, timeout=30) as response:
    listing = json.load(response)
for entry in listing:
    if entry.get("type") == "file" and entry.get("download_url"):
        destination = os.path.join(demo, entry["name"])
        if not (os.path.exists(destination) and os.path.getsize(destination) == entry.get("size")):
            urllib.request.urlretrieve(entry["download_url"], destination)
print("  " + ", ".join(sorted(os.listdir(demo))))
PYEOF

step "permissions: the clone readable by everyone, writable only by its owner"
chmod -R a+rX,go-w "$REPO"

step "check"
"$PY" - <<'PYEOF'
import importlib.metadata as md
import torch
for pkg in ("abismal", "torch", "torchref", "jupyterlab", "ipywidgets"):
    print(f"  {pkg} {md.version(pkg)}")
from abismal.gui.components import ArgparseGUI
ArgparseGUI()
print("  GUI builds")
print(f"  torch built for CUDA {torch.version.cuda}; GPU usable here: {torch.cuda.is_available()}")
PYEOF
"$ENV/bin/abismal" --help >/dev/null && echo "  abismal CLI runs"
if command -v nvidia-smi >/dev/null 2>&1 && ! "$PY" -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
    driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
    echo
    echo "  This node has a GPU (its driver supports CUDA $driver_cuda) but torch cannot use it:"
    echo "  the driver is older than torch's CUDA build. Rerun with an older build, e.g."
    echo "    $0 --torch-index https://download.pytorch.org/whl/cu128"
fi
echo
echo "Installed in $INSTALL. Users paste this into the Jupyter app's custom-script field:"
echo "  source $HERE/demo_setup.sh"
