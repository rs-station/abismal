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

# Install TensorFlow Probability
source <(curl -s https://raw.githubusercontent.com/rs-station/careless/main/install-tfp.sh)

# Reactivate to update cuda paths
conda activate $ENVNAME

# Install abismal
pip install abismal

