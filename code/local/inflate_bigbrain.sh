#!/bin/bash
# Inflates bigbrain surface

LOCAL_CODE_PATH=$(realpath $(dirname "$0"))
cd "$(dirname "$0")/../../"

module load freesurfer/7.1
mris_inflate src/tpl-bigbrain_hemi-L_desc-mid.surf.gii src_local/tpl-bigbrain_hemi-L_desc-mid.surf.inflate.gii
mris_inflate src/tpl-bigbrain_hemi-R_desc-mid.surf.gii src_local/tpl-bigbrain_hemi-R_desc-mid.surf.inflate.gii