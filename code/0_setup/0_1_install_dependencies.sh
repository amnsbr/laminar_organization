#!/bin/bash
cd "$(dirname "$0")"
#> Install python libraries
pip install -r requirements.txt

#> Install HCP workbench
if ! command -v wb_command &> /dev/null
then
    echo "HCP workbench could not be found"
    mkdir '../../tools'
    wget -nc -P '../../tools' \
        'https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip'
    unzip -n '../../tools/workbench-linux64-v1.5.0.zip' -d '../../tools/'
    wb_path=$(realpath "../../tools/workbench")
    PATH="${PATH}:${wb_path}"
fi

#> Install BigBrainWarp
if ! command -v bigbrainwarp &> /dev/null
then
    echo "BigBrainWarp could not be found"
    mkdir '../../tools'
    git clone https://github.com/caseypaquola/BigBrainWarp.git '../../tools/BigBrainWarp'
    bbwDir=$(realpath "../../tools/BigBrainWarp")
    PATH="${PATH}:${bbwDir}"
fi