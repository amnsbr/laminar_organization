import os
import pandas as pd
import numpy as np
import nilearn.surface
import cortex.polyutils

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', '..', 'data')

def create_curvature_surf_map():
    """
    Creates the map of curvature for the bigbrain surface
    using pycortex

    TODO: Needs validation
    """
    for hem in ['L', 'R']:
        vertices, faces = nilearn.surface.load_surf_mesh(
            os.path.join(
                DATA_DIR, 'surface',
                f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                )
        )
        surface = cortex.polyutils.Surface(vertices, faces)
        curvature = surface.mean_curvature()
        curvature_filepath = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-{hem}_desc-mean_curvature.npy'
            )
        np.save(
            curvature_filepath,
            curvature
        )

def calculate_covariates():
    """
    Wrapper for functions that are used for creating the covariates
    """
    create_curvature_surf_map()

if __name__=='__main__':
    calculate_covariates()
