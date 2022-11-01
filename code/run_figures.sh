#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"

for nb_file in 'Fig_connectivity.ipynb' #'Fig_hierarchy.ipynb' #'Fig_ltcg.ipynb' # #'Fig_MPC_ctypes.ipynb' #'Fig_1.ipynb'
do
    eval "$(conda shell.bash hook)" && \
    conda activate ${PROJECT_DIR}/laminar_gradients_conda && \
    ${PROJECT_DIR}/laminar_gradients_conda/bin/jupyter nbconvert --to notebook --inplace --execute ${PROJECT_DIR}/code/${nb_file}
done