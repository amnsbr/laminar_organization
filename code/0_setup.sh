#!/bin/bash
cd "$(dirname "$0")/.."
#> Install python libraries
python3 -m "venv" laminar_gradients_env
source laminar_gradients_env/bin/activate && \
laminar_gradients_env/bin/pip install --upgrade pip && \
laminar_gradients_env/bin/pip install -r code/requirements.txt

mkdir 'tools'
cd 'tools'
if ! [ -f 'bigbrainwarp.simg' ]; then
    singularity pull docker://caseypaquola/bigbrainwarp
fi