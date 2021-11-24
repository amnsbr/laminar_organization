#!/bin/bash
# Creates separate density profiles for each layer

echo "Creating separate density profiles for each layer"

LOCAL_CODE_PATH=$(realpath $(dirname "$0"))
cd "$(dirname "$0")/../../src"
SRC_PATH=$(realpath .)
TOOLS_PATH=$(realpath ../tools)
SRC_LOCAL_PATH=$(realpath ../src_local)
mkdir density_profiles

# Pull CIVET's singularity if it doesn't exist
if ! [ -f "${TOOLS_PATH}/civet-2.1.1.simg" ]; then
    singularity pull docker://mcin/civet:2.1.1
else
    echo "Singularity image exists"
fi

for hem in 'left' 'right'; do
    for (( num=1; num<=6; num++ )); do
        let above_num=num-1
        echo "Upper layer: ${SRC_PATH}/layer${above_num}_${hem}_327680.obj"
        echo "Lower layer: ${SRC_PATH}/layer${num}_${hem}_327680.obj"
        echo "Output dir: ${SRC_PATH}/density_profiles/${hem}-layer${num}"
        bash "$LOCAL_CODE_PATH"/sample_intensity_profiles_singularity.sh \
            --in_vol "${SRC_PATH}/full16_100um_optbal.mnc" \
            --upper_surf "${SRC_PATH}/layer${above_num}_${hem}_327680.obj" \
            --lower_surf "${SRC_PATH}/layer${num}_${hem}_327680.obj" \
            --num_surf 10 \
            --wd "${SRC_PATH}/density_profiles/${hem}-layer${num}/"
        cp "${SRC_PATH}/density_profiles/${hem}-layer${num}/profiles.npz" "${SRC_LOCAL_PATH}/density_profile_hemi-${hem}_layer-${num}_nsurf-10.npz"
    done
done