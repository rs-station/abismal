# Sort the dependencies
source <(curl -s https://raw.githubusercontent.com/rs-station/abismal/main/pre-install.sh)

# Refinement (--torchref-pdb) runs on CPU, so install the CPU build of PyTorch
# before the extras pull in the CUDA one. See README.md -- removing the CUDA
# build afterwards deletes shared libraries that TensorFlow needs.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Extras compose: one install, one resolution pass.
pip install "abismal[cuda,torchref]"
