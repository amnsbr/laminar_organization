#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"

for nb_file in 'Fig_ltcg' 'Fig_hierarchy' 'Fig_connectivity' 'Fig_development' 'Fig_S_ltcg.ipynb' 'Fig_S_hierarchy' 'Fig_S_connectivity'
do
    eval "$(conda shell.bash hook)" && \
    conda activate ${PROJECT_DIR}/laminar_gradients_conda && \
    ${PROJECT_DIR}/laminar_gradients_conda/bin/jupyter nbconvert --to notebook --inplace --execute --allow-errors ${PROJECT_DIR}/code/${nb_file}.ipynb
done