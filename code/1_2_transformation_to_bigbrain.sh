#!/bin/bash
# Transforms parcellations to the bigbrain space
#> Prepare the environment for bigbrainwarp to properly work
cd "../src"
export bbwDir=$(realpath "../tools/BigBrainWarp")
export wb_path=$(realpath "../tools/workbench")
export PATH="${PATH}:${wb_path}:${bbwDir}"
source "../laminar_gradients_env/bin/activate"
python() {
    "../laminar_gradients_env/bin/python" "$@"
}
export -f python
#> Do the transformation
fsaverage_to_bigbrain=('sjh.annot' 'economo.annot')
for suffix in "${fsaverage_to_bigbrain[@]}"
do
    if [ -f "../data/parcellation/tpl-bigbrain_hemi-L_desc-${suffix/.annot/_parcellation}.label.gii" ]; then
        echo "${suffix} already transformed to bigbrain space"
    else
        echo "Transforming ${suffix} to bigbrain space"
        bigbrainwarp \
        --in_lh "lh_${suffix}" \
        --in_rh "rh_${suffix}" \
        --in_space fsaverage \
        --out_space bigbrain \
        --out_den 32 \
        --interp nearest \
        --desc "${suffix/.annot/_parcellation}" \
        --wd "../data/parcellation"
    fi
done