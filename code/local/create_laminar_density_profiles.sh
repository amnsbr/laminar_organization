#!/bin/bash
# Creates separate density profiles for each layer

echo "Creating separate density profiles for each layer"

LOCAL_CODE_PATH=$(realpath $(dirname "$0"))
cd "$(dirname "$0")/../../src"
SRC_PATH=$(realpath .)
TOOLS_PATH=$(realpath ../tools)
SRC_LOCAL_PATH=$(realpath ../src_local)

mkdir $SRC_LOCAL_PATH
cd $SRC_LOCAL_PATH
mkdir density_profiles

# Download BigBrain volume
if ! [ -f "full16_100um_optbal.mnc" ]; then
    echo "Downloading BigBrain volume"
    wget -t 0 "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Volumes/Histological_Space/mnc/full16_100um_optbal.mnc"
else
    echo "Found BigBrain volume"
fi

# Download layer boundaries
if ! [ -f "layer0_left_327680.obj.gz" ]; then
    echo "Downloading layer boundaries"
    wget -t 0 -r -nH --cut-dirs=6 --no-parent --reject="index.html*" "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/Layer_Segmentation/3D_Surfaces/PLoSBiology2020/MNI-obj/"
else
    echo "Found layer boundaries"
fi

# Pull CIVET's singularity if it doesn't exist
if ! [ -f "${TOOLS_PATH}/civet-2.1.1.simg" ]; then
    cd $TOOLS_PATH
    singularity pull docker://mcin/civet:2.1.1
    cd $SRC_LOCAL_PATH
else
    echo "Singularity image exists"
fi

for hem in 'left' 'right'; do
    for (( num=1; num<=6; num++ )); do
        let above_num=num-1
        echo "Upper layer: ${SRC_LOCAL_PATH}/layer${above_num}_${hem}_327680.obj"
        echo "Lower layer: ${SRC_LOCAL_PATH}/layer${num}_${hem}_327680.obj"
        echo "Output dir: ${SRC_LOCAL_PATH}/density_profiles/${hem}-layer${num}"
        bash "$LOCAL_CODE_PATH"/sample_intensity_profiles_singularity.sh \
            --in_vol "${SRC_LOCAL_PATH}/full16_100um_optbal.mnc" \
            --upper_surf "${SRC_LOCAL_PATH}/layer${above_num}_${hem}_327680.obj" \
            --lower_surf "${SRC_LOCAL_PATH}/layer${num}_${hem}_327680.obj" \
            --num_surf 10 \
            --wd "${SRC_LOCAL_PATH}/density_profiles/${hem}-layer${num}/"
        cp "${SRC_LOCAL_PATH}/density_profiles/${hem}-layer${num}/profiles.npz" "${SRC_PATH}/density_profile_hemi-${hem}_layer-${num}_nsurf-10.npz"
    done
done