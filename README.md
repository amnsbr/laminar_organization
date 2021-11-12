# Laminar Structure Similarity Gradients

## Dataset structure

- External tools (incl. BigBrainWarp and HCP Workbench) will be downloaded to `tools/` by the script `coce/0_setup.sh`
- External inputs (i.e. building blocks from other sources) will be downloaded to `src/` by the script `code/1_1_download_src.py`.
- Local inputs are located in `src_local/` and will be copied to `src/` by the script `code/1_1_download_src.py`
- All code is located in `code/`.
  - HTCondor .submit files corresponding to bash scripts are located in `code/htcondor/`
- Important / processed input files in addition to the output files are located in `data/`

Space needed by all the data and tools: `3.5 GB`

## Requirements
Singularity is the only requirement that is not installed by `0_setup.sh`.
