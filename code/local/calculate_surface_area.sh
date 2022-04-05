#!/usr/bin/bash
# Wrapper for calculate_surface_area.py to make it compatible with htcondor

# Specify directories
cd "$(dirname "$0")/../../src"
SRC_PATH=$(realpath .)
TOOLS_PATH=$(realpath ../tools)

# Pull CIVET's singularity if it doesn't exist
if ! [ -f "${TOOLS_PATH}/civet-2.1.1.simg" ]; then
    cd $TOOLS_PATH
    singularity pull docker://mcin/civet:2.1.1
    cd $SRC_PATH

eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/laminar_gradients_conda && \
${PROJECT_DIR}/laminar_gradients_conda/bin/python ${HOME}/laminar_gradients/code/local/calculate_surface_area.py