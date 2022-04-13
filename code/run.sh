#!/bin/bash
# Wrapper for code/run.py to make it compatible with htcondor
# (otherwise the virtual environment is not activated properly)
cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"

eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/laminar_gradients_conda && \
${PROJECT_DIR}/laminar_gradients_conda/bin/python ${PROJECT_DIR}/code/run.py