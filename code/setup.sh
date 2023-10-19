#!/bin/bash
cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"

# Create conda environment to specify Python version
if [ ! -d ${PROJECT_DIR}/env ]; then
    echo "Installing environment in ${PROJECT_DIR}/env"
    conda create -y -p ${PROJECT_DIR}/env python=3.9.12
fi

# But install packages through pip (it's faster & more suitable for docker!)
eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env && \
conda install pip -y && \
${PROJECT_DIR}/env/bin/pip install -r ${PROJECT_DIR}/code/requirements.txt