#!/bin/bash
# Transforms parcellations to the bigbrain space
cd "$(dirname "$0")"
cd "../../src"
export bbwDir="../tools/BigBrainWarp"

for suffix in ('sjh.annot' 'economo.annot')
do
    echo "Transforming ${suffix} to bigbrain space"
    bigbrainwarp \
    --in_lh "lh_${suffix}" \
    --in_rh "rh_${suffix}" \
    --in_space fsaverage \
    --out_space bigbrain \
    --out_den 32 \
    --interp nearest \
    --desc "${suffix/.annot/_parcellation}" \
    --wd "../data/parcellations"
done