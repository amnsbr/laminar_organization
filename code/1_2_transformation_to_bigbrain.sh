#!/bin/bash
# Transforms parcellations to the bigbrain space
cd "$(dirname "$0")/../src"
SRC_PATH=$(realpath .)
fsaverage_to_bigbrain=('sjh.annot' 'economo.annot')
for suffix in "${fsaverage_to_bigbrain[@]}"
do
    if [ -f "../data/parcellation/tpl-bigbrain_hemi-L_ssdesc-${suffix/.annot/_parcellation}.label.gii" ]; then
        echo "${suffix} already transformed to bigbrain space"
    else
        echo "Transforming ${suffix} to bigbrain space (singularity)"
        #> Run bigbrainwarp with singularity
        # `singularity run` is not used because of problems with environment
        # variables. Instead I have used `singularity exec` to 1) run init.sh
        # sctip, 2) cd to the correct path (SRC_PATH), and 3) run bigbrainwarp 
        singularity exec --cleanenv "../tools/bigbrainwarp.simg" \
        /bin/bash -c \
        "source /BigBrainWarp/scripts/init.sh && \
        cd ${SRC_PATH} && \
        /BigBrainWarp/bigbrainwarp \
        --in_lh 'lh_${suffix}' \
        --in_rh 'rh_${suffix}' \
        --in_space fsaverage \
        --out_space bigbrain \
        --out_den 32 \
        --interp nearest \
        --desc '${suffix/.annot/_parcellation}' \
        --wd '../data/parcellation'"
    fi
done