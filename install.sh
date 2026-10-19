
# Sort the dependencies
source <(curl -s https://raw.githubusercontent.com/rs-station/abismal/main/pre-install.sh)

# torch comes with abismal. The default Linux wheel bundles CUDA, so a GPU is
# used with no further setup; on a machine without one, add
#   --extra-index-url https://download.pytorch.org/whl/cpu
# to take the much smaller CPU build instead. See README.md.
pip install "abismal[torchref]"
