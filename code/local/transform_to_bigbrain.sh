#!/bin/bash

# Specify directories
cd "$(dirname "$0")/../../src"
SRC_PATH=$(realpath .)
TOOLS_PATH=$(realpath ../tools)

# Download bigbrainwarp
if ! [ -f "${TOOLS_PATH}/bigbrainwarp.simg" ]; then
    cd ${TOOLS_PATH}
    singularity pull docker://caseypaquola/bigbrainwarp
    cd ${SRC_PATH}
fi


fsaverage_parcellations=('sjh.annot' 'economo.annot' 'schaefer200.annot' 'schaefer400.annot' 'schaefer1000.annot' 'aparc.annot' 'mmp1.annot' 'brodmann.label.gii')
for suffix in "${fsaverage_parcellations[@]}"
do
    desc=${suffix/.annot/_parcellation}
    desc=${desc/.label.gii/_parcellation}
    if [ -f "${SRC_PATH}/tpl-bigbrain_hemi-L_desc-${desc}.label.gii" ]; then
        echo "${desc} already transformed to bigbrain space"
    else
        echo "Transforming ${desc} to bigbrain space (singularity)"
        #> Run bigbrainwarp with singularity
        # `singularity run` is not used because of problems with environment
        # variables. Instead I have used `singularity exec` to 1) run init.sh
        # sctip, 2) cd to the correct path (SRC_PATH), and 3) run bigbrainwarp 
        singularity exec --cleanenv "${TOOLS_PATH}/bigbrainwarp.simg" \
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
        --desc ${desc} \
        --wd ${SRC_PATH}"
    fi
done

if [ -f "${SRC_PATH}/tpl-bigbrain_hemi-L_desc-hcp1200_myelinmap.label.gii" ]; then
    echo "HCP 1200 myelination map already transformed to bigbrain space"
else
    echo "Transforming HCP 1200 myelination map to bigbrain space (singularity)"
    singularity exec --cleanenv "${TOOLS_PATH}/bigbrainwarp.simg" \
    /bin/bash -c \
    "source /BigBrainWarp/scripts/init.sh && \
    cd ${SRC_PATH} && \
    /BigBrainWarp/bigbrainwarp \
    --in_lh 'source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-L_feature.func.gii' \
    --in_rh 'source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-R_feature.func.gii' \
    --in_space fs_LR \
    --out_space bigbrain \
    --interp nearest \
    --desc hcp1200_myelinmap \
    --wd ${SRC_PATH}"
fi