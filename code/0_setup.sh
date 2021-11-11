#!/bin/bash
cd "$(dirname "$0")/.."
#> Install python libraries
python3 -m "venv" laminar_gradients_env --upgrade-deps
source laminar_gradients_env/bin/activate &\
laminar_gradients_env/bin/pip install -r code/requirements.txt

#> Install HCP workbench
if ! command -v wb_command &> /dev/null; then
    echo "HCP workbench could not be found"
    mkdir 'tools'
    if ! [ -f 'tools/workbench-linux64-v1.5.0.zip' ]; then
        echo "Downloading HCP workbench v 1.5.0"
        wget -nc -P 'tools' \
            'https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip'
        unzip -n 'tools/workbench-linux64-v1.5.0.zip' -d 'tools/'
    fi
    wb_path=$(realpath "tools/workbench/bin_linux64")
    PATH="${PATH}:${wb_path}"
fi

#> Install BigBrainWarp
if ! command -v bigbrainwarp &> /dev/null; then
    echo "BigBrainWarp could not be found"
    mkdir 'tools'
    if ! [ -d 'tools/BigBrainWarp' ]; then
        echo "Downloading BigBrainWarp"
        git clone https://github.com/caseypaquola/BigBrainWarp.git 'tools/BigBrainWarp'
        #> checkout the specific commit at the time of analysis
        cd "tools/BigBrainWarp"
        git checkout 1c11effaf5461dec128e7ed83f42716c4461a88d
        cd "$(dirname "$0")/.."
    fi
    bbwDir=$(realpath "tools/BigBrainWarp")
    PATH="${PATH}:${bbwDir}"
fi