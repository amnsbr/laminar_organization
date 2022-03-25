#!/usr/bin/bash
# Wrapper for code/fetch_ahba.py to make it compatible with htcondor
# (otherwise the virtual environment is not activated properly)
source ${HOME}/laminar_gradients/laminar_gradients_env/bin/activate & \
${HOME}/laminar_gradients/laminar_gradients_env/bin/pip install abagen==0.1.3 & \
${HOME}/laminar_gradients/laminar_gradients_env/bin/python ${HOME}/laminar_gradients/code/local/fetch_ahba.py