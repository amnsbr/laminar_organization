#!/bin/bash
cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"
## TODO: conda installation (4.12.0)
## TODO: install wb_command (1.5.0)

# Install conda and pip dependencies based on environment.yml
if [ ! -d ${PROJECT_DIR}/env ]; then
    echo "Installing conda env in ${PROJECT_DIR}/env"
    conda create -y -p ${PROJECT_DIR}/env python=3.9.12
fi

# Install most packages through pip (it's faster & more suitable for docker!)
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env && \
conda install pip -y && \
${PROJECT_DIR}/env/bin/pip install -r ${PROJECT_DIR}/code/requirements.txt

# Install mesalib, vtk and brainspace separately to make sure
# the osmesa build of vtk is installed and brainspace plotting works
# on remote servers
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env && \
conda install mesalib==21.2.5 -y --channel conda-forge --override-channels --freeze-installed
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env && \
conda install vtk==9.1.0 -y --channel conda-forge --override-channels --freeze-installed && \
${PROJECT_DIR}/env/bin/pip install brainspace==0.1.3


# Install other dependencies
mkdir ${PROJECT_DIR}/tools
cd ${PROJECT_DIR}/tools
git clone --depth 1 --branch 1.1.3 https://github.com/MICA-MNI/ENIGMA.git
cd ENIGMA
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env && \
${PROJECT_DIR}/env/bin/pip install .

# reinstall numpy using conda
# when installed using pip, numpy (1.23.4) is incompatible
# with pycortex (1.2.3) and importing pycortex throws an error. 
# This is the best fix I could find!
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env && \
${PROJECT_DIR}/env/bin/pip uninstall -y numpy && \
conda install -y --channel conda-forge "numpy==1.23.4"

