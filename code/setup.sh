#!/bin/bash
cd "$(dirname "$0")/.."
#> Install python libraries
python3 -m "venv" laminar_gradients_env
source laminar_gradients_env/bin/activate && \
laminar_gradients_env/bin/pip install --upgrade pip && \
laminar_gradients_env/bin/pip install -r code/requirements.txt

#> Install non-pip tools
mkdir 'tools'
cd 'tools'
git clone https://github.com/MICA-MNI/ENIGMA.git
cd ENIGMA
source ../../laminar_gradients_env/bin/activate && \
../../laminar_gradients_env/bin/python setup.py install

