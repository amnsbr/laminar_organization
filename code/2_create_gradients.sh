#!/usr/bin/bash
# Wrapper for code/2_create_gradients.py to make it compatible with htcondor
# (otherwise the virtual environment is not activated properly)
source ${HOME}/laminar_gradients_datalad/laminar_gradients_env/bin/activate & \
python ${HOME}/laminar_gradients_datalad/code/2_create_gradients.py