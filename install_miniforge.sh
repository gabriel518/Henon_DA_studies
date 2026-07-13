set -e                                                    

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"                                                                                                     
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ./miniforge -f
source miniforge/bin/activate
python -m pip install -r requirements.txt

# prebuild Xsuite's compiled kernels (the xsuite-prebuild command comes with
# the xsuite meta-package in requirements.txt). Without this the kernels are
# built lazily on the first build_tracker() call -- a few extra seconds on
# script 01c's first run.
xsuite-prebuild

# optional: GPU tracking (uncomment the cupy line in requirements.txt too).
# cupy's pip wheel bundles its own CUDA runtime, so a system-wide
# cudatoolkit is normally NOT needed -- only install it if you need to
# compile custom CUDA kernels from source.
# conda install mamba -n base -c conda-forge
# mamba install cudatoolkit=11.8.0