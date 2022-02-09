"""
Calculates wm and pia surface area of the BigBrain surface using CIVET
singularity image. This will be used for estimating relative laminar
volume from relative laminar thickness based on the curvature (~
the relative surface area of wm vs pia at each vertex)

The code is adapted from https://github.com/kwagstyl/surface_tools
"""
import numpy as np
import nilearn.surface
import subprocess
import argparse
import os
import copy
import sys
sys.path.append(os.getcwd() + '/..')
import helpers

abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
SRC_DIR = os.path.abspath(os.path.join(cwd, '..', '..', 'src'))
TOOLS_DIR = os.path.abspath(os.path.join(cwd, '..', '..', 'tools'))
SRC_LOCAL_DIR = os.path.abspath(os.path.join(cwd, '..', '..', 'src_local'))

fwhm = 0
civet_singularity = os.path.join(TOOLS_DIR, 'civet-2.1.1.simg')


def calculate_area(surfname,fwhm, civet_singularity=None):
    """calculate and smooth surface area using CIVET"""
    tmpdir='/tmp/' + str(np.random.randint(1000))
    os.mkdir(tmpdir)
    if civet_singularity:
        civet_singularity_prefix = f"singularity exec {civet_singularity} "
    else:
        civet_singularity_prefix = ""
    try:
        subprocess.call(civet_singularity_prefix + "depth_potential -area_voronoi " + surfname + " " +os.path.join(tmpdir,"tmp_area.txt"),shell=True)
        if fwhm ==0:
            area=np.loadtxt(os.path.join(tmpdir,"tmp_area.txt"))
        else:
            subprocess.call(civet_singularity_prefix + "depth_potential -smooth " + str(fwhm) + " " + os.path.join(tmpdir,"tmp_area.txt ") + surfname + " "+os.path.join(tmpdir,"sm_area.txt"),shell=True)
            area=np.loadtxt(os.path.join(tmpdir,"sm_area.txt"))
        subprocess.call("rm -r "+tmpdir,shell=True)
    except OSError:
        print("depth_potential not found, please install CIVET tools or replace with alternative area calculation/data smoothing")
        return 0
    return area   


for hem in ['L', 'R']:
    #> download the requirements
    os.chdir(SRC_LOCAL_DIR)
    for surf in ['white', 'pial']:
        helpers.download(
        f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-{surf}.obj')

    gray = os.path.join(SRC_LOCAL_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial.obj')
    white = os.path.join(SRC_LOCAL_DIR, f'tpl-bigbrain_hemi-{hem}_desc-white.obj')

    wm_vertexareas = calculate_area(white, fwhm, civet_singularity=civet_singularity)
    pia_vertexareas = calculate_area(gray, fwhm, civet_singularity=civet_singularity)

    np.save(
        os.path.join(
            SRC_DIR,
            f'tpl-bigbrain_hemi-{hem}_desc-white.area.npy'
        ), 
        wm_vertexareas
    )
    np.save(
        os.path.join(
            SRC_DIR,
            f'tpl-bigbrain_hemi-{hem}_desc-pial.area.npy'
        ),
        pia_vertexareas
    )