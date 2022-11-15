#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_DIR="$(realpath .)"
nb_files="$(ls ${PROJECT_DIR}/code/figures/*.ipynb)" 
for nb_file in $nb_files
do
    eval "$(conda shell.bash hook)" && \
    conda activate ${PROJECT_DIR}/env && \
    ${PROJECT_DIR}/env/bin/jupyter nbconvert --to notebook --inplace --execute --allow-errors $nb_file
done