#!/bin/bash
# Wrapper for all the functions that load the data
code_dir=$(dirname "$0")
cd $code_dir # to have reliable relative paths

#> 1. Downloads parcellations, spaces and laminar thickness data into 'src' folder 
#  and copies those that need no preprocessing to the 'data' folder
echo "-----------Downloading the source-------------"
source "../laminar_gradients_env/bin/activate" &\ 
"../laminar_gradients_env/bin/python" "1_1_download_src.py"

#> 2. Transform atlases/parcellations in fsaverage space to bigbrain space 
#  using bigbrainwarp
echo "-----------Transforming atlases/parcellations to bigbrain space-------------"
bash "1_2_transformation_to_bigbrain.sh"

#> 3. Create masks of bigbrain space including agranular and dysgranular region
echo "-----------Creating masks of bigbrain space including agranular and dysgranular region-----------"
source "../laminar_gradients_env/bin/activate" &\
"../laminar_gradients_env/bin/python" 1_3_create_adysgranular_mask.py -p sjh

#> 4. Calculate the covariates (curvature and geodesic distance)
echo "-----------Calculating covariates-----------"
source "../laminar_gradients_env/bin/activate" &\
"../laminar_gradients_env/bin/python" 1_4_calculate_covariates.py