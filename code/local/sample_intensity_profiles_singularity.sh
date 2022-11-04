#!/bin/bash
# The original script is a part of bigbrainwarp which can be found in
# https://github.com/caseypaquola/BigBrainWarp/blob/master/scripts/sample_intensity_profiles.sh
# This is slightly modified to work with CIVET on singularity
#>>> start of the original script
# generates staining intensity profiles given a volume and matched surface files

#---------------- FUNCTION: HELP ----------------#
help() {
echo -e "
\033[38;5;141mCOMMAND:\033[0m
   $(basename "$0")
\033[38;5;141mREQUIRED ARGUMENTS:\033[0m
\t\033[38;5;197m-in_vol\033[0m 	      		: input volume to be sampled. Must be .mnc
\t\033[38;5;197m-upper_surf\033[0m 	      	: upper surface. Must be aligned to the volume and an .obj
\t\033[38;5;197m-lower_surf\033[0m              : lower surface. Must be aligned to the volume and an .obj
\t\033[38;5;197m-num_surf\033[0m              	: number of surfaces to generate
\t\033[38;5;197m-wd\033[0m 	              	: Path to a working directory, where data will be output

Casey Paquola, MNI, MICA Lab, 2021
https://bigbrainwarp.readthedocs.io/
"
}

# Create VARIABLES
for arg in "$@"
do
  case "$arg" in
  -h|-help)
    help
    exit 1
  ;;
  --in_vol)
    in_vol=$2
    shift;shift
  ;;
  --wd)
    wd=$2
    shift;shift
  ;;
  --upper_surf)
    upper_surf=$2
    shift;shift
  ;;
  --lower_surf)
    lower_surf=$2
    shift;shift
  ;;
  --num_surf)
    num_surf=$2
    shift;shift
  ;;
  --civet_singualrity_path)
    civet_singualrity_path=$2
    shift;shift
  ;;
  -*)
    echo "Unknown option ${2}"
    help
    exit 1
  ;;
    esac
done

# create wd
if [[ ! -d "$wd" ]]; then
  mkdir "$wd"
fi

# specify the pathways
LOCAL_CODE_PATH=$(realpath $(dirname "$0"))
TOOLS_PATH=$(realpath ../tools)
CIVET_PATH="$TOOLS_PATH"/civet-2.1.1.simg

# pull surface tools repo, if not already contained in scripts
if [[ ! -d "$TOOLS_PATH"/surface_tools/ ]] ; then
	cd $TOOLS_PATH
	git clone https://github.com/amnsbr/surface_tools
fi
cd "$TOOLS_PATH"/surface_tools/equivolumetric_surfaces/

"${TOOLS_PATH}/../env/bin/python" generate_equivolumetric_surfaces.py "$upper_surf" "$lower_surf" "$num_surf" "$wd"/ --civet_singularity "$CIVET_PATH"
x=$(ls -t "$wd"/*.obj) # orders my time created
for n in $(seq 1 1 "$num_surf") ; do
	echo "$n"
	which_surf=$(sed -n "$n"p <<< "$x")
	# make numbering from upper to lower
	let "nd = "$num_surf" - "$n""
	singularity exec "$CIVET_PATH" volume_object_evaluate "$in_vol" "$which_surf" "$wd"/"$nd".txt
done

cd "$LOCAL_CODE_PATH"
"${TOOLS_PATH}/../env/bin/python" compile_profiles.py "$wd" "$num_surf"