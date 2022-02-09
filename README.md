# Gradients of Microstructural Covariance in the BigBrain

## Dataset structure
- Source files are located in `src/`
- All code is located in `code/`
    - `setup.sh` creates the virtual environment and installs python dependencies
    - `run.sh` is a wrapper for `run.py` which runs all the analyses on `src/` files and stores them in `output/`
    - `code/local/` includes codes that cannot easily be included in the automated pipeline (`run.py`) because of their dependencies (e.g. FreeSurfer or docker/singularity containers)
    - `code/htcondor/` includes HTCondor `.submit` files corresponding to each bash script