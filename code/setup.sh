#!/bin/bash
cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"
## TODO: conda installation
## TODO: install wb_command

# Install conda and pip dependencies based on environment.yml
echo "Installing conda env in ${PROJECT_DIR}/laminar_gradients_conda"
conda env create -f ${PROJECT_DIR}/code/environment.yml -p ${PROJECT_DIR}/laminar_gradients_conda

# Install mesalib, vtk and brainspace separately to make sure
# the osmesa build of vtk is installed and brainspace plotting works
# on remote servers
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/laminar_gradients_conda && \
conda install mesalib -y --channel conda-forge --override-channels --freeze-installed && \
conda install vtk -y --channel conda-forge --override-channels --freeze-installed && \
${PROJECT_DIR}/laminar_gradients_conda/bin/pip install brainspace==0.1.3


# Install other dependencies
mkdir ${PROJECT_DIR}/tools
cd ${PROJECT_DIR}/tools
git clone https://github.com/MICA-MNI/ENIGMA.git
cd ENIGMA
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/laminar_gradients_conda && \
${PROJECT_DIR}/laminar_gradients_conda/bin/pip install .