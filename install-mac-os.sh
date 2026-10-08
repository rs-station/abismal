# abismal on macOS. torch's macOS wheels include the Metal (MPS) backend, and
# abismal selects it automatically on Apple silicon, so there is nothing
# GPU-specific to install. `abismal --list-devices` shows what it found.
ENVNAME=abismal
PY_VERSION=3.12

conda activate base

result=$(conda create -n $ENVNAME python=$PY_VERSION 3>&2 2>&1 1>&3)

echo $result
if [[ $result == *"CondaSystemExit"* ]]; then
    echo "User aborted anaconda env creation. Exiting... "
    return
fi

conda activate $ENVNAME
pip install --upgrade pip

# The pins match pre-install.sh, so adding the torchref extra later does not
# overwrite conda-built packages. See "About the version pins" in README.md.
conda install -c conda-forge -y dials "numpy<2.4" "pandas<2.4" "scipy<1.18"

# Install abismal
pip install abismal
