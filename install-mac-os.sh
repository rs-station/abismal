ENVNAME=abismal
PY_VERSION=3.11
TFP_VERSION=0.24.0
TF_VERSION=2.16.1

conda activate base

result=$(conda create -n $ENVNAME python=$PY_VERSION 3>&2 2>&1 1>&3)

echo $result
if [[ $result == *"CondaSystemExit"* ]]; then
    echo "User aborted anaconda env creation. Exiting... "
    return
fi

conda activate $ENVNAME
pip install --upgrade pip

conda install -c conda-forge dials -y
pip install tensorflow==$TF_VERSION
pip install tensorflow-probability[tf]==$TFP_VERSION
pip install tensorflow-metal

# Install rs version with parallel stream loading
pip install git+https://github.com/rs-station/reciprocalspaceship@parstream
pip install ray

# Install abismal
pip install abismal

