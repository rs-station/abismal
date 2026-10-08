# Create the abismal environment: DIALS from conda-forge, everything else on pip.
#
# The pins keep the conda solve inside the numpy/pandas/scipy windows that
# torchref requires, so pip does not have to overwrite conda-built packages
# afterwards. See the "About the version pins" section of README.md.
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

conda install -c conda-forge -y dials "numpy<2.4" "pandas<2.4" "scipy<1.18"
