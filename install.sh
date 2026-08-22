# Sort the dependencies
source <(curl -s https://raw.githubusercontent.com/rs-station/abismal/main/pre-install.sh)

# Install abismal release. Add the torchref extra for per-epoch refinement
# (--torchref-pdb); it pulls PyTorch into the same environment, so no separate
# torchref environment is needed.
pip install "abismal[torchref]"
