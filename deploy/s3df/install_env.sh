#!/usr/bin/env bash
# Build the shared abismal environment for the S3DF Jupyter demo.
#
# Run once per release by someone with write access to the install location; users
# only ever read from it. Everything lands under PREFIX:
#
#   PREFIX/bin/micromamba          the package manager used to build the env
#   PREFIX/env/                    python, torch (CUDA), abismal[gui], torchref, jupyterlab
#   PREFIX/share/demo/hewl/        the lysozyme example data and its reference files
#   PREFIX/activate.sh             what the per-user pre-launch script sources
#   PREFIX/demo_setup.sh           a copy of the pre-launch script, for offline use
#   PREFIX/abismal_s3df_demo.ipynb the demo notebook users get a copy of
#
# Usage:
#   deploy/s3df/install_env.sh PREFIX [--ref GIT_REF] [--torch-index URL] [--with-dials]
#
#   --ref          abismal git ref to install (default: torch)
#   --torch-index  PyTorch wheel index; pick the newest CUDA the S3DF GPU driver supports
#                  (default: https://download.pytorch.org/whl/cu128)
#   --with-dials   also install DIALS/dxtbx from conda-forge, needed for .expt/.refl
#                  inputs (several GB; .mtz and CrystFEL .stream inputs never need it)
#
# Needs internet access (GitHub, conda-forge, PyPI, download.pytorch.org): run it on a
# login or interactive node that has it. Safe to rerun; it updates in place.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

PREFIX=""
REF="torch"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
WITH_DIALS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        --with-dials) WITH_DIALS=1; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "unknown option $1" >&2; usage 1 ;;
        *) PREFIX="$1"; shift ;;
    esac
done
[ -n "$PREFIX" ] || usage 1
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="$PREFIX/env"
PY="$ENV/bin/python"
step() { printf '\n== %s\n' "$*"; }

step "micromamba"
if [ ! -x "$PREFIX/bin/micromamba" ]; then
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$PREFIX" bin/micromamba
fi
MAMBA="$PREFIX/bin/micromamba"
export MAMBA_ROOT_PREFIX="$PREFIX/mamba"

step "python environment at $ENV"
if [ ! -x "$PY" ]; then
    "$MAMBA" create -y -q -p "$ENV" -c conda-forge python=3.12 pip git
fi
if [ "$WITH_DIALS" = 1 ]; then
    step "dials (conda-forge)"
    "$MAMBA" install -y -q -p "$ENV" -c conda-forge dials
fi

step "torch from $TORCH_INDEX"
"$PY" -m pip install -q --no-cache-dir --upgrade torch --index-url "$TORCH_INDEX"

step "abismal[gui] @ $REF, torchref, jupyterlab"
"$PY" -m pip install -q --no-cache-dir --upgrade \
    "abismal[gui] @ git+https://github.com/rs-station/abismal@${REF}" torchref jupyterlab ipykernel

step "example data"
DEMO="$PREFIX/share/demo/hewl"
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

step "activate.sh, demo_setup.sh, notebook"
cat > "$PREFIX/activate.sh" <<EOF
# Put the shared abismal environment first on PATH. Sourced by demo_setup.sh.
export ABISMAL_S3DF_PREFIX="$PREFIX"
export PATH="$ENV/bin:\$PATH"
EOF
cp "$HERE/demo_setup.sh" "$HERE/abismal_s3df_demo.ipynb" "$PREFIX/"
"$PY" -c "import abismal, subprocess; print(abismal.__file__)" >/dev/null
git -C "$HERE" rev-parse HEAD > "$PREFIX/DEPLOYED_FROM" 2>/dev/null || echo "$REF" > "$PREFIX/DEPLOYED_FROM"

step "permissions: readable by everyone, writable only by the owner"
chmod -R a+rX,go-w "$PREFIX"

step "smoke test"
"$PY" - <<'PYEOF'
import importlib.metadata as md
import torch
print("  torch", torch.__version__, "CUDA build", torch.version.cuda,
      "| GPU visible here:", torch.cuda.is_available())
for pkg in ("abismal", "torchref", "jupyterlab", "ipywidgets"):
    print(f"  {pkg} {md.version(pkg)}")
from abismal.gui.components import ArgparseGUI
ArgparseGUI()
print("  GUI builds")
PYEOF
"$ENV/bin/abismal" --help >/dev/null && echo "  abismal CLI runs"
echo
echo "Installed at $PREFIX. Users source it with:"
echo "  source $PREFIX/demo_setup.sh"
