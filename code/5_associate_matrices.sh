#!/usr/bin/bash
# Wrapper for code/2_create_gradients.py to make it compatible with htcondor
# (otherwise the virtual environment is not activated properly)
source ${HOME}/laminar_gradients/laminar_gradients_env/bin/activate & \
${HOME}/laminar_gradients/laminar_gradients_env/bin/python ${HOME}/laminar_gradients/code/5_associate_matrices.py