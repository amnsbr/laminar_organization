import urllib.request
import shutil
from urllib.parse import urlparse
import os
import glob
import gc
from ftplib import FTP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import brainspace.mesh, brainspace.plotting, brainspace.null_models
import brainsmash.mapgen
import enigmatoolbox.permutation_testing
import nilearn.surface
import nilearn.plotting
from nilearn.input_data import NiftiLabelsMasker
import nibabel
import statsmodels.api as sm
import scipy.io
import abagen
import subprocess
import requests
import time

import matrices
import datasets

# specify the directories and constants
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
SPIN_BATCHES_DIR = os.path.join(SRC_DIR, 'spin_batches')
os.makedirs(SPIN_BATCHES_DIR, exist_ok=True)

MIDLINE_PARCELS = {
    'schaefer200': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'schaefer400': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'schaefer1000': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'sjh': [0],
    'aparc': ['L_unknown', 'None'],
    'mmp1': ['???'],
    'brodmann': ['???'],
    'economo': ['L_unknown', 'R_unknown'],
    'aal': ['None'],
    'M132': ['???', 'MedialWall']
}
###### Loading data ######
def download(url, file_name=None, copy_to=None, overwrite=False):
    """
    Download the file from `url` and save it locally under `file_name`.
    Also creates a copy in 'copy_to'
    """
    if not file_name:
        file_name = os.path.basename(urlparse(url).path)
    print('Downloading', file_name, end=' ')
    if (os.path.exists(file_name) and not overwrite):
        print(">> already exists")
    else:
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(">> done")
        
    if copy_to:
        if not os.path.exists(copy_to):
            shutil.copyfile(file_name, copy_to)

def download_bigbrain_ftp(ftp_dir, ftp_filename, out_filename=None, copy_to=None):
    """
    Downloads a file from bigbrain ftp
    """
    # Connect
    ftp = FTP('bigbrain.loris.ca')
    ftp.login()
    # Go to dir
    ftp.cwd(ftp_dir)
    # Determine filename
    if not out_filename:
        out_filename = ftp_filename
    print('Downloading', out_filename, end=' ')
    # Download
    if not os.path.exists(out_filename):
        with open(out_filename, 'wb') as file_obj:
            ftp.retrbinary(f'RETR {ftp_filename}', file_obj.write)
        print(">> done")
    else:
        print(">> already exists")
    # Copy to another folder if needed
    if copy_to:
        if not os.path.exists(copy_to):
            shutil.copyfile(out_filename, copy_to)

###### Data manipulation ######
def parcellate(surface_data, parcellation_name, averaging_method='median', 
               na_midline=True, space='bigbrain', na_ratio_cutoff=1.0, hemi=None):
    """
    Parcellates `surface data` using `parcellation` and by taking the
    median or mean (specified via `averaging_method`) of the vertices within each parcel.

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) 
        n_vertices x n_features surface data of L and/or R hemispheres
    parcellation_name: (str)
    averaging_method: (str) 
        Method of averaging over vertices within a parcel. Default: 'median'
        - 'median'
        - 'mean'
        - None (will return groupby object)
    na_midline: (bool) 
        make midline vertices NaN
    space: (str)
        - 'bigbrain' (default)
        - 'fsaverage'
        - 'yerkes'
    na_ratio_cutoff: (float)
        parcels with na_ratio beyond this cutoff will be set to NaN
    hemi: (None | str)

    Returns
    ---------
    parcellated_data: (pd.DataFrame or dict of pd.DataFrame) 
        n_parcels x n_features for data of L and/or R hemispheres
    """
    if isinstance(surface_data, dict):
        parcellated_data = {}
        for hemi in ['L', 'R']:
            parcellated_data[hemi] = parcellate(surface_data[hemi], parcellation_name, averaging_method, 
                na_midline, space, na_ratio_cutoff, hemi=hemi)
    elif isinstance(surface_data, np.ndarray):
        if space in ['bigbrain','fsaverage']:
            # determine if the surface data is downsampled
            if hemi is None:
                is_downsampled = (surface_data.shape[0] == datasets.N_VERTICES_HEM_BB_ICO5*2)
            else:
                is_downsampled = (surface_data.shape[0] == datasets.N_VERTICES_HEM_BB_ICO5)
        else:
            is_downsampled = False
        # load parcellation map
        labeled_parcellation_map = datasets.load_parcellation_map(
            parcellation_name, 
            concatenate=(hemi is None),
            downsampled=is_downsampled,
            space=space)
        if hemi is not None:
            labeled_parcellation_map = labeled_parcellation_map[hemi]
        # parcellate
        parcellated_vertices = (
            pd.DataFrame(surface_data, index=labeled_parcellation_map)
        )
        if na_midline:
            for midline_parcel in MIDLINE_PARCELS.get(parcellation_name, []):
                if midline_parcel in parcellated_vertices.index:
                    parcellated_vertices.loc[midline_parcel] = np.NaN
        parcellated_vertices_groupby = (parcellated_vertices
            .reset_index(drop=False)
            .groupby('index')
        )
        # operate on groupby object if needed
        if averaging_method is None:
            parcellated_data = parcellated_vertices_groupby
        else:
            if averaging_method == 'median':
                parcellated_data = parcellated_vertices_groupby.median()
            elif averaging_method == 'mean':
                parcellated_data = parcellated_vertices_groupby.mean()
            # remove parcels with many NaN vertices
            if na_ratio_cutoff < 1:
                high_na_parcels = []
                for parcel, vertices_data in parcellated_vertices_groupby:
                    if parcellated_vertices.reset_index().loc[vertices_data.index][0].isna().mean() > na_ratio_cutoff:
                        high_na_parcels.append(parcel)
                parcellated_data.loc[high_na_parcels] = np.NaN
    return parcellated_data

def parcellate_volumetric(img_path, parcellation_name):
    """
    Parcellated volumetric image
    """
    # prepare the parcellation masker
    # Warning: Background label is by default set to
    # 0. Make sure this is the case for all the parcellation
    # maps and zero corresponds to background / midline
    masker = NiftiLabelsMasker(
        os.path.join(
            SRC_DIR, 
            f'tpl-MNI152_desc-{parcellation_name}_parcellation.nii.gz'
            ), 
        strategy='mean',
        resampling_target='data',
        background_label=0)
    parcellated_data = masker.fit_transform(img_path).flatten()
    # add labels of the parcels
    parcellated_data = pd.Series(
        parcellated_data,
        index = datasets.load_volumetric_parcel_labels(parcellation_name)
    )
    return parcellated_data

def concat_hemispheres(parcellated_data, dropna=False):
    """
    Concatenates the parcellated data of L and R hemispheres

    Parameters
    ----------
    parcellated_data: (dict of pd.DataFrame) n_parcels x 6 for laminar data of L and R hemispheres
    dropna: (bool) remove parcels with no data. Default: False

    Returns
    ----------
    concat_data: (pd.DataFrame) n_parcels*2 x 6
    """
    # concatenate hemispheres and drop NaN if needed
    concat_data = (pd.concat(
        [
            parcellated_data['L'], 
            parcellated_data['R']
        ],
        axis=0)
        # take the average values in case parcels are repeated in L and R
        # e.g. for midline parcels
        .reset_index(drop=False) 
        .groupby('index').mean()
        )
    if dropna:
        concat_data = concat_data.dropna()
    return concat_data

def downsample(surface_data, space='bigbrain'):
    """
    Downsamples `surface data` in bigbrain ico7 space to ico5 by treating the mapping
    between ico7 and ico5 (previously calculated in `local/downsample_bb.m`) as a
    pseudo-parcellation, where each parcel consists of the central vertex shared between ico7
    and ico5 ('bb_downsample') + its neighbors (found in 'nn_bb'). Then the average of
    each parcel is calculated and the "parcels" (or ico5 central vertices) are ordered
    to match ico5 surface.

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) n_vertices_ico7 x n_features
    space: (str)
        - bigbrain
        - fsaverage

    Returns
    ---------
    downsampled_surface_data: (np.ndarray or dict of np.ndarray) n_vertices_ico5 x n_features
    """
    concat_output = False
    if isinstance(surface_data, np.ndarray):
        surface_data = {
            'L': surface_data[:datasets.N_VERTICES_HEM_BB],
            'R': surface_data[datasets.N_VERTICES_HEM_BB:],
        }
        concat_output = True
    downsampled_surface_data = {}
    for hem in ['L', 'R']:
        if space == 'bigbrain':
            # load the downsampled surface from matlab results
            mat = scipy.io.loadmat(
                os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                    )
            )
            # load the "parcellation map" of downsampled bigbrain surface
            #  indicating the index for the center vertex of each parcel
            #  in the original ico7 (320k) BB surface
            downsample_parcellation = mat['nn_bb'][0, :]-1 #1-indexing to 0-indexing
            # then "parcellate" the ico7 data to ico5 by taking
            #  the average of center vertices and their neighbors
            #  indicated in 'nn_bb'
            downsampled = (
                pd.DataFrame(surface_data[hem], index=downsample_parcellation)
                .reset_index(drop=False)
                .groupby('index')
                .mean()
            )
            # if there is a single NaN value around a downsampled vertex
            # set that vertex to NaN
            any_nan = (
                pd.DataFrame(surface_data[hem], index=downsample_parcellation)
                .isna()
                .reset_index(drop=False)
                .groupby('index')
                .sum()
                .any(axis=1)
            )
            downsampled.loc[any_nan] = np.NaN
            # load correctly ordered indices of center vertices
            #  and reorder downsampled data to the correct order alligned with ico5 surface
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            downsampled_surface_data[hem] = downsampled.loc[bb_downsample_indices].values
        elif space == 'fsaverage':
            downsampled_surface_data[hem] = surface_data[hem][:datasets.N_VERTICES_HEM_BB_ICO5]
    if concat_output:
        return np.concatenate([
            downsampled_surface_data['L'],
            downsampled_surface_data['R']
            ], axis=0)
    else:
        return downsampled_surface_data

def upsample(surface_data, space='bigbrain'):
    """
    (Pseudo-)upsampling of surface_data from ico5 to ico7. This is useful
    for data that is originally created in ico5 space (e.g. the unparcellated
    gradients).

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) n_vertices_ico5 x n_features
    space: (str)

    Returns
    ---------
    upsampled_surface_data: (np.ndarray or dict of np.ndarray) n_vertices_ico7 x n_features
    """
    if space=='fsaverage':
        raise NotImplementedError
    concat_output = False
    if isinstance(surface_data, np.ndarray):
        surface_data = {
            'L': surface_data[:datasets.N_VERTICES_HEM_BB_ICO5],
            'R': surface_data[datasets.N_VERTICES_HEM_BB_ICO5:],
        }
        concat_output = True
    upsampled_surface_data = {}
    for hem in ['L', 'R']:
        # load the downsampled surface from matlab results
        mat = scipy.io.loadmat(
            os.path.join(
                SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
        )
        # load the "parcellation map" of downsampled bigbrain surface
        #  indicating the index for the center vertex of each parcel
        #  in the original ico7 (320k) BB surface
        downsample_parcellation = mat['nn_bb'][0, :]-1 #1-indexing to 0-indexing
        # load correctly ordered indices of center vertices in the ico7 space
        # (mapping of vertex indices from ico5 to ico7)
        bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
        # upsample the surface by "deparcellating" ico5 surface to ico7
        # (see `deparcellate` function to better understand how it works)
        upsampled_surface_data[hem] = (
            pd.DataFrame(surface_data[hem], index=bb_downsample_indices)
            .loc[downsample_parcellation]
        )
    if concat_output:
        return np.concatenate([
            upsampled_surface_data['L'],
            upsampled_surface_data['R']
            ], axis=0)
    else:
        return upsampled_surface_data

def regress_out_surf_covariates(input_surface_data, cov_surface_data, sig_only=False, renormalize=False):
    """
    Fits `cov_surface_data` to `input_surface_data` and return the residual.

    Parameters
    ----------
    input_surface_data: (np.ndarray) n_vertices x n_cols input surface data
    cov_surface_data: (np.ndarray) n_vertices covariate surface data
    sig_only: (bool) do the regression only if the correlation is significant
    renormalize: (bool) used when input_surface_data is normalized/unnormalized laminar thickness
                        whether to normalize the residual laminar thicknesses to a sum of 1 to
                        get relative laminar thickness corrected for cov_surface_data (e.g. curvature)
    Return
    --------
    cleaned_surface_data: (np.ndarray) n_vertices x n_cols cleaned surface data
    """
    cleaned_surface_data = input_surface_data.copy()
    mask = (np.isnan(input_surface_data).sum(axis=1) == 0)
    for col_idx in range(input_surface_data.shape[1]):
        print(f"col {col_idx}")
        # read and reshape y
        y = input_surface_data[mask, col_idx].reshape(-1, 1)
        # demean and store mean (instead of fitting intercept)
        y_mean = y.mean()
        y_demean = y - y_mean
        # convert covariate shape to n_vert * 1 if there's only one
        X = cov_surface_data.reshape(-1, 1)
        X = X[mask, :]
        assert X.shape[0] == y.shape[0]
        # model fitting
        lr = sm.OLS(y_demean, X).fit()
        print(f'pval {lr.pvalues[0]} {"*" if lr.pvalues[0] < 0.05 else ""}')
        if (lr.pvalues[0] < 0.05) or (not sig_only):
            cleaned_surface_data[mask, col_idx] = lr.resid + y_mean
        else:
            cleaned_surface_data[mask, col_idx] = y[:, 0]
    if renormalize:
        cleaned_surface_data /= cleaned_surface_data.sum(axis=1, keepdims=True)    
    return cleaned_surface_data

def get_gd_disc(mesh, vertex_number, radius):
    """
    Get a disc of vertices that are `radius` mm apart from the seed
    `vertex_number` on the `mesh`

    Parameters
    ----------
    mesh: (str)
        Path to the cortical mesh
    vertex_number: (int)
    radius: (int)
        in mm

    Returns
    ------
    gd_mask: (np.ndarray:bool)
    """
    tmp_fname = f'/tmp/gd_{vertex_number}_{np.random.random(1)[0] * 1e8:.0f}.shape.gii'
    subprocess.run(f"""
    wb_command -surface-geodesic-distance \
        '{mesh}' \
        {vertex_number} \
        '{tmp_fname}' \
        -limit {radius}
    """, shell=True)
    # read the output gifti file into a numpy array
    gd = nibabel.load(tmp_fname).agg_data()
    gd_mask = gd >= 0
    return gd_mask

def disc_smooth(surface_data, smooth_disc_radius, approach='euclidean'):
    """
    Smoothes the surface data (in ico5) using discs created on the
    inflated surface with `smooth_disc_radius` around each vertex.
    The smoothing is done uniformly

    Parameters
    ----------
    surface_data: (dict of np.ndarray) n_vert_ico5 x n_features
    smooth_disc_radius: (int)
    approach: (str)
        - euclidean
        - geodesic: is very computationally costly

    Returns
    -------
    smoothed_surface_data: (dict of np.ndarray) n_vert_ico5 x n_features
    discs: (dict np.ndarray) n_vert_ico5 x n_vert_ico5
    """
    assert surface_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB_ICO5
    inflated_mesh_paths = datasets.load_mesh_paths('inflated', downsampled=True)
    discs = {}
    smoothed_surface_data = {}
    for hem in ['L', 'R']:
        # load inflated mesh
        inflated_mesh = nilearn.surface.load_surf_mesh(inflated_mesh_paths[hem])
        if approach == 'euclidean':
            # calculate Euclidean distance of all pairs of vertices
            ed_matrix = scipy.spatial.distance_matrix(inflated_mesh.coordinates, inflated_mesh.coordinates)
            # create a matrix of discs for each row/column seed vertex
            discs[hem] = ed_matrix < smooth_disc_radius
        # loop through vertices and calculate the smoothed value by
        # taking the average of the disc surrounding it
        smoothed_surface_data[hem] = np.zeros_like(surface_data[hem])
        for vertex in range(datasets.N_VERTICES_HEM_BB_ICO5):
            if np.isnan(surface_data[hem][vertex, :]).any():
                # only do smoothing if the seed is not NaN
                smoothed_surface_data[hem][vertex, :] = np.NaN
            else:
                if approach == 'euclidean':
                    disc = discs[hem][vertex, :].astype('bool')
                else:
                    disc = get_gd_disc(inflated_mesh_paths[hem], vertex, smooth_disc_radius)
                smoothed_surface_data[hem][vertex, :] = np.nanmean(surface_data[hem][disc, :], axis=0)
    return smoothed_surface_data, discs

def deparcellate(parcellated_data, parcellation_name, downsampled=False, space='bigbrain'):
    """
    Project the parcellated data to surface vertices while handling empty parcels
    (parcels that are not in the parcellated data but are in the parcellation map)

    Parameters
    ----------
    parcellated_data: (pd.DataFrame | pd.Series) n_parcels x n_features
    parcellation_name: (str)
    downsampled: (bool)
    space: (str)
        - 'bigbrain'
        - 'fsaverage'

    Returns
    -------
    surface_map: (np.ndarray) n_vertices [both hemispheres] x n_features
    """
    # load concatenated parcellation map
    concat_parcellation_map = datasets.load_parcellation_map(
        parcellation_name, concatenate=True, downsampled=downsampled, space=space)
    # load dummy parcellated data covering the whole brain
    if space in ['bigbrain', 'fsaverage']:
        if downsampled:
            dummy_surf_data = np.zeros(datasets.N_VERTICES_HEM_BB_ICO5 * 2)
        else:
            dummy_surf_data = np.zeros(datasets.N_VERTICES_HEM_BB * 2)
    elif space in ['yerkes', 'fs_LR']:
        dummy_surf_data = np.zeros(datasets.N_VERTICES_HEM_FSLR * 2)
    parcellated_dummy = parcellate(
        dummy_surf_data,
        parcellation_name,
        space=space)
    all_parcels = parcellated_dummy.index.to_series().rename('parcel')
    # create a dataframe including all parcels, where invalid parcels are NaN
    #   (this is necessary to be able to project it to the parcellation)
    labeled_parcellated_data = pd.concat(
        [
            parcellated_data,
            all_parcels
        ], axis=1).set_index('parcel')
    # get the surface map by indexing the parcellated map at parcellation labels
    surface_map = labeled_parcellated_data.loc[concat_parcellation_map].values # shape: vertex X gradient
    # TODO: convert it back to DataFrame or Series with original col names
    return surface_map

def get_valid_parcels(parcellation_name, exc_regions, thr=0.5, downsampled=True):
    """
    Get the valid parcels of `parcellation_name` that are 
    not midline and have less than `thr` of their vertices 
    in the `exc_regions`.

    Parameters
    ----------
    parcellation_name: (str)
    exc_regions: (str)
    thr: (float)
    downsampled: (bool)

    Returns
    -------
    valid_parcels: (pd.Index)
    """
    if exc_regions is None:
        if downsampled:
            exc_mask = np.zeros(datasets.N_VERTICES_HEM_BB_ICO5*2).astype('bool')
        else:
            exc_mask = np.zeros(datasets.N_VERTICES_HEM_BB*2).astype('bool')
    else:
        exc_mask = datasets.load_exc_masks(exc_regions, concatenate=True, downsampled=downsampled)
    parcellated_exc_mask = (
        # calculate the fraction of vertices in each parcel that
        # are within the exc_mask
        parcellate(exc_mask, parcellation_name, 'mean')[0]
        # drop the midline and threshold
        .dropna()
        >= thr
    )
    return parcellated_exc_mask.loc[~parcellated_exc_mask].index

def get_hem_parcels(parcellation_name, limit_to_parcels=None, space='bigbrain'):
    """
    Get parcels belonging to each hemisphere from all the parcels
    or the list `limit_to_parcels`

    Parameters
    --------
    parcellation_name: (str)
    limit_to_parcels: (list or None)
        get the intersection of the parcels with this list of parcels in each
        hemisphere. useful for splitting an existing matrix into hemispheres

    Returns
    -------
    parcels: (dict of list)
    """
    if space in ['bigbrain', 'fsaverage']:
        dummy = {
            'L': np.zeros(datasets.N_VERTICES_HEM_BB),
            'R': np.zeros(datasets.N_VERTICES_HEM_BB)
            }
    else:
        dummy = {
            'L': np.zeros(datasets.N_VERTICES_HEM_FSLR),
            'R': np.zeros(datasets.N_VERTICES_HEM_FSLR)
            }
    parcellated_dummy = parcellate(dummy, parcellation_name, space=space)
    parcels = {}
    for hem in ['L', 'R']:
        # get the index of parcels that are not NA (midline)
        parcels[hem] = parcellated_dummy[hem].dropna().index.tolist()
        if limit_to_parcels is not None:
            # get its intersection with the list of parcels
            parcels[hem] = list(set(limit_to_parcels) & set(parcels[hem]))
    return parcels


def get_parcel_center_indices(parcellation_name, space='bigbrain', kind='orig', downsampled=False):
    """
    Gets the center of parcels and returns their index
    in BigBrain inflated surface

    Parameters
    ---------
    parcellation_name: (str)
    space: (str)
        see datasets.load_mesh_paths
    kind: (str)
        see datasets.load_mesh_paths
    downsampled: (bool)

    Returns
    -------
    centers: (dict of pd.Series)

    Credit
    ------
    Based on "geoDistMapper.py" from micapipe/functions but using
    inflated surface so that euclidean distance is closer to geodesic
    Original Credit:
    # Translated from matlab:
    # Original script by Boris Bernhardt and modified by Casey Paquola
    # Translated to python by Jessica Royer
    """
    centers = {}
    for hem in ['L', 'R']:
        print(f"Finding parcel centers in {hem} hemisphere")
        out_path = os.path.join(
            SRC_DIR,
            f'tpl-{space}_{kind}_hemi-{hem}_downsampled-{downsampled}_parc-{parcellation_name}_desc-centers.csv'
            )
        if os.path.exists(out_path):
            centers[hem] = pd.read_csv(out_path, index_col='parcel').iloc[:, 0]
            continue
        # load inflated surf
        surf_path = datasets.load_mesh_paths(kind, space=space, downsampled=downsampled)[hem]
        vertices = nilearn.surface.load_surf_mesh(surf_path).coordinates           
        parc = datasets.load_parcellation_map(parcellation_name, False, downsampled=downsampled, space=space)[hem]
        # loop through parcels and find the centers
        centers[hem] = pd.Series(0, index=np.unique(parc))        
        for parcel_name in centers[hem].index:
            this_parc = np.where(parc == parcel_name)[0]
            if this_parc.size == 1: # e.g. L_unknown in aparc
                centers[hem].loc[parcel_name] = this_parc[0]
            else:
                distances = scipy.spatial.distance.pdist(np.squeeze(vertices[this_parc,:]), 'euclidean') # Returns condensed matrix of distances
                distancesSq = scipy.spatial.distance.squareform(distances) # convert to square form
                sumDist = np.sum(distancesSq, axis = 1) # sum distance across columns
                index = np.where(sumDist == np.min(sumDist)) # minimum sum distance index
                centers[hem].loc[parcel_name] = this_parc[index[0][0]]
        centers[hem].to_csv(out_path, index_label='parcel')
    return centers


def get_parcel_boundaries(parcellation_name, space='bigbrain'):
    """
    Finds the boundaries of parcels on the surface mesh

    Parameters
    ---------
    parcellation_name: (str)
    space: (str)
        see datasets.load_mesh_paths
    
    Returns
    -------
    boundaries: (np.ndarray)
        (n_vertices, ) binary array with 1 indicating boundary vertices and 0 otherwise
    """
    boundaries = {}
    mesh_paths = datasets.load_mesh_paths('inflated', space=space, downsampled=False)
    for hemi in ['L', 'R']:
        mesh = nilearn.surface.load_surf_mesh(mesh_paths[hemi])
        parcellation_map = datasets.load_parcellation_map(parcellation_name, concatenate=False, load_indices=True)[hemi]
        boundaries[hemi] = np.zeros_like(parcellation_map)
        for parc in np.unique(parcellation_map):
            parc_mask = (parcellation_map==parc)
            boundary_faces = (parc_mask[mesh.faces].sum(axis=1) == 1)
            parc_mask_boundary_verts = list(
                set(mesh.faces[boundary_faces].flatten()) & \
                set(np.where(parc_mask)[0])
            )
            boundaries[hemi][parc_mask_boundary_verts] = 1
    return np.concatenate([boundaries['L'], boundaries['R']])


###### Plotting ######
def make_colorbar(vmin, vmax, cmap=None, bins=None, orientation='vertical', figsize=None):
    """
    Plots a colorbar

    Parameters
    ---------
    vmin, vmax: (float)
    cmap: (str or `matplotlib.colors.Colormap`)
    bins: (int)
        if specified will plot a categorical cmap
    orientation: (str)
        - 'vertical'
        - 'horizontal'
    figsize: (tuple)

    Returns
    -------
    fig: (`matplotlib.figure.Figure`)
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.linspace(vmin, vmax, 100).reshape(10, 10), cmap=plt.cm.get_cmap(cmap, bins))
    fig.gca().set_visible(False)
    divider = make_axes_locatable(ax)
    if orientation == 'horizontal':
        cax = divider.append_axes("bottom", size="10%", pad=0.05)
    else:
        cax = divider.append_axes("left", size="10%", pad=0.05)
    fig.colorbar(im, cax=cax, ticks=np.array([vmin, vmax]), orientation=orientation)
    cax.yaxis.tick_left()
    cax.xaxis.tick_bottom()
    return fig

def plot_matrix(matrix, outpath=None, cmap="rocket", vrange=(0.025, 0.975), vrange_value=None, **kwargs):
    """
    Plot the matrix as heatmap

    Parameters
    ----------
    matrix: (np.ndarray) square matrix or horizontally concatenated
        square matrices
    cmap: (str or colormap objects) colormap recognizable by seaborn
    vrange: 
        - (tuple): vmin and vmax as percentiles (for whole range put (0, 1))
        - 'sym'
    vrange_value: (None | tuple)
        vrange as actual values instead of percentiles
    """
    n_square_matrices = matrix.shape[1] // matrix.shape[0]
    if vrange_value is not None:
        vmin = vrange_value[0]
        vmax = vrange_value[1]
    else:
        if vrange == 'sym':
            vmin = min(np.nanmin(matrix.flatten()), -np.nanmax(matrix.flatten()))
            if vmin < 0:
                vmax = -vmin
            else:
                vmax = np.nanmax(matrix.flatten())
        else:
            vmin = np.nanquantile(matrix.flatten(),vrange[0])
            vmax = np.nanquantile(matrix.flatten(),vrange[1])
    fig, ax = plt.subplots(figsize=(4 * n_square_matrices,4), dpi=192)
    sns.heatmap(
        matrix,
        vmin=vmin, vmax=vmax,
        cmap=cmap, cbar=False,
        ax=ax, **kwargs)
    ax.axis('off')
    clbar_fig = make_colorbar(
        vmin=vmin, vmax=vmax,
        cmap=cmap, orientation='horizontal'
        )
    if outpath:
        fig.tight_layout()
        fig.savefig(outpath, dpi=192)
        clbar_fig.savefig(outpath+'_clbar', dpi=192)


def plot_surface(surface_data, filename=None, space='bigbrain', inflate=True, 
        plot_downsampled=True, layout_style='row', cmap='viridis',
        toolbox='brainspace', vrange=None, cbar=False, nan_color=(0.75, 0.75, 0.75, 1),
        **plotter_kwargs):
    """
    Plots the surface data with medial and lateral views of both hemispheres

    Parameters
    ----------
    surface_data: (np.ndarray) (n_vert,) 
    filename: (str | None) 
        path to output without .png
    space: (str)
        - 'bigbrain'
        - 'fsaverage'
        - 'fs_LR'
        - 'yerkes'
    inflate: (bool) 
        whether to plot the inflated surface
    plot_downsampled: (bool) 
        whether to plot the ico5 vs ico7 surface
    layout_style:
        - row: left-lateral, left-medial, right-medial, right-lateral
        - grid: lateral views on the top and medial views on the bottom
    cmap: (str)
    toolbox: (str)
        - brainspace
        - nilearn
    vrange: (None | 'sym' | tuple)
    cbar: (bool)
        create color bar (in a separate figure)
    **plotter_kwargs: keyword arguments specific to brainspace or nilearn plotters

    Returns
    -------
    (matplotlib.figure.Figure or path-like str)
    """
    if vrange is None:        
        vrange = (np.nanmin(surface_data), np.nanmax(surface_data))
    elif vrange == 'sym':
        vmin = min(np.nanmin(surface_data), -np.nanmax(surface_data))
        vrange = (vmin, -vmin)
    # split surface to L and R and make sure the shape is correct
    if space in ['bigbrain', 'fsaverage']:
        if surface_data.shape[0] == datasets.N_VERTICES_HEM_BB * 2:
            lh_surface_data = surface_data[:datasets.N_VERTICES_HEM_BB]
            rh_surface_data = surface_data[datasets.N_VERTICES_HEM_BB:]
        elif surface_data.shape[0] == datasets.N_VERTICES_HEM_BB_ICO5 * 2:
            lh_surface_data = surface_data[:datasets.N_VERTICES_HEM_BB_ICO5]
            rh_surface_data = surface_data[datasets.N_VERTICES_HEM_BB_ICO5:]
            plot_downsampled = True
        else:
            raise TypeError("Wrong surface data dimensions")
    elif space in ['fs_LR', 'yerkes']: 
        # only 32k version is supported
        lh_surface_data = surface_data[:datasets.N_VERTICES_HEM_FSLR]
        rh_surface_data = surface_data[datasets.N_VERTICES_HEM_FSLR:]
    surface_data = {'L': lh_surface_data, 'R': rh_surface_data}
    # specify the mesh and downsample the data if needed
    if plot_downsampled & (space in ['bigbrain', 'fsaverage']):
        if (surface_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB):
            surface_data = downsample(surface_data, space=space)
    if inflate:
        kind = 'inflated'
    else:
        kind = 'orig'
    mesh_paths = datasets.load_mesh_paths(kind=kind, space=space, downsampled=plot_downsampled)
    # plot the colorbar
    if cbar:
        clbar_fig = make_colorbar(*vrange, cmap)
        if filename:
            clbar_fig.tight_layout()
            clbar_fig.savefig(filename+'_clbar.png', dpi=192)
    # plot the surfaces
    if toolbox == 'brainspace':
        plotter_kwargs['nan_color'] = nan_color
        return _plot_brainspace(surface_data, mesh_paths, filename, layout_style, cmap, vrange, **plotter_kwargs)
    else:
        return _plot_nilearn(surface_data, mesh_paths, filename, layout_style, cmap, vrange, **plotter_kwargs)

def _plot_brainspace(surface_data, mesh_paths, filename, layout_style, cmap, vrange, **plotter_kwargs):
    """
    Plots `surface_data` on `mesh_paths` using nilearn

    Note: To run this on remote server vtk should be installed with
    mesabuild as follows:
    > conda config --add channels conda-forge
    > conda install mesalib --channel conda-forge --override-channels -freeze-installed
    > conda install vtk --channel conda-forge --override-channels -freeze-installed
    # if conda tries to install a build of vtk that does not start with osmesa_* force this build using:
    > conda install vtk==9.1.0=osmesa_py39h8ab48e2_107 --channel conda-forge --override-channels -freeze-installed
    """
    # rejoin the hemispheres
    surface_data = np.concatenate([surface_data['L'], surface_data['R']]).flatten()
    # load bigbrain surfaces
    lh_surf = brainspace.mesh.mesh_io.read_surface(mesh_paths['L'])
    rh_surf = brainspace.mesh.mesh_io.read_surface(mesh_paths['R'])
    # read surface data files and concatenate L and R
    if filename:
        screenshot = True
        embed_nb = False
        filename += '.png'
    else: # TODO: this is not working
        screenshot = False
        embed_nb = True
    if layout_style == 'row':
        size = (1600, 400)
        zoom = 1.2
    else:
        size = (900, 500)
        zoom = 1.8
    return brainspace.plotting.surface_plotting.plot_hemispheres(
        lh_surf, rh_surf, 
        surface_data,
        layout_style = layout_style,
        cmap = cmap, color_range=vrange,
        size=size, zoom=zoom,
        interactive=False, embed_nb=embed_nb,
        screenshot=screenshot, filename=filename, 
        transparent_bg=True,
        **plotter_kwargs
        )

def _plot_nilearn(surface_data, mesh_paths, filename, layout_style, cmap, vrange, **plotter_kwargs):
    """
    Plots `surface_data` on `mesh_paths` using nilearn
    """
    # initialize the figures
    if layout_style == 'row':
        figure, axes = plt.subplots(1, 4, figsize=(24, 5), subplot_kw={'projection': '3d'})
    elif layout_style == 'grid':
        figure, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        # reorder axes so that lateral views are on top, matching the order of axes
        #  in the horizontal layout
        axes = np.array([axes[0, 0], axes[1, 0], axes[1, 1], axes[0, 1]])
    curr_ax_idx = 0
    for hemi in ['left', 'right']:
        mesh_path = mesh_paths[hemi[0].upper()]
        # specify the view order
        if hemi == 'left':
            views_order = ['lateral', 'medial']
        else:
            views_order = ['medial', 'lateral']
        # plot
        for view in views_order:
            nilearn.plotting.plot_surf(
                mesh_path,
                surface_data[hemi[0].upper()],
                hemi=hemi, view=view, axes=axes[curr_ax_idx],
                cmap=cmap,
                vmin=vrange[0],
                vmax=vrange[1],
                **plotter_kwargs,
                )
            curr_ax_idx += 1
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    if filename:
        figure.savefig(filename, dpi=192)
    else:
        return figure

def run_csea_developmental(gene_list, fdr=True, plot=True, **plot_kwargs):
    """
    Runs CSEA tool developmental enrichment on the gene list 
    using http://genetics.wustl.edu/jdlab/csea-tool-2/

    Parameters
    ---------
    gene_list: (list of str)
    fdr: (bool)
    plot: (bool)
    **plot_kwargs: see plot_csea_res_table

    Returns
    -------
    csea_res: (pd.DataFrame)
        CSEA developmental enrichment output table
    """
    time.sleep(3) # to avoid breaking their servers with too many requests
    res = requests.post('http://genetics.wustl.edu/jdlab-fisher/cgi-bin/seaBrainRegion.cgi', 
                {
                    'symbols': ' '.join(gene_list), 
                    'Candidate gene list from': 'Human'
                })
    if res.status_code != 200:
        print("Request failed")
        return
    else:
        csea_res = pd.read_html(res.text)[0]
        csea_res.columns = csea_res.iloc[0, :]
        csea_res = csea_res.drop(0).set_index('Brain Regions and Development and P-Values')
        if fdr:
            csea_res = csea_res.applymap(lambda c: (c.split('(')[-1][:-1])).astype('float')
        else:
            csea_res = csea_res.applymap(lambda c: (c.split('(')[0])).astype('float')
        if plot:
            plot_csea_res_table(csea_res, **plot_kwargs)
        return csea_res

def plot_csea_res_table(csea_res, pSI_thresh=0.05, color='red', cmap=None):
    """
    Plots CSEA tool results in a heatmap

    csea_res: (pd.DataFrmae)
        CSEA developmental enrichment output table
    pSI_thresh: (float)
    color: (str)
        for barplot
    cmap: (str)
        for heatmap
    """
    sns.set_style('ticks')
    csea_res = csea_res.loc[:, [str(pSI_thresh)]]
    csea_res['Structure'] = csea_res.index.to_series().apply(lambda s: s.split('.')[0].strip())
    csea_res['Stage'] = csea_res.index.to_series().apply(lambda s: ' '.join(s.split('.')[1:]))
    csea_res['Stage'] = csea_res['Stage'].str.strip()
    csea_res = csea_res.reset_index(drop=True).set_index(['Stage', 'Structure'])[str(pSI_thresh)].unstack()
    stages_order = [
        'Early Fetal', 'Early Mid Fetal', 'Late Mid Fetal', 'Late Fetal',
        'Neotal Early Infancy', 'Late Infancy', 'Early Childhood',
        'Middle Late Childhood', 'Adolescence', 'Young Adulthood'
    ]
    regions_order = [
        'Cortex', 'Thalamus', 'Cerebellum', 'Striatum', 'Hippocampus', 'Amygdala'
    ]
    csea_res = csea_res.loc[stages_order, regions_order]
    csea_res[csea_res > 0.05] = np.NaN
    fig, ax = plt.subplots(1, figsize=(8,4), dpi=192)
    if cmap is None:
        cmap = f'{color.title()}s'
    sns.heatmap(-np.log(csea_res).T, cmap=cmap, vmin=-np.log(0.05), vmax=20, ax=ax)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    fig, ax = plt.subplots(1, figsize=(8,2), dpi=192)
    ax.bar(x = csea_res.index, height = -np.log(csea_res).T.sum(axis=0), color=color, alpha=1)
    ax.set_ylim([0, 60])
    ax.set_xticklabels(stages_order, rotation=90);

def plot_stacked_bar(df, colors):
    """
    Plots stacked bar plot.

    Parameters
    ---------
    df: (pd.DataFrame)
        columns indicate height and are ordered from bottom to top
        index indicates location of each stacked bar along x axis
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=192)
    ax.bar(
        x = df.index,
        height = df.iloc[:, 0],
        width = 1,
        color=colors[0],
        )
    for idx in range(1, df.shape[1]):
        ax.bar(
            x = df.index,
            height = df.iloc[:, idx],
            bottom = df.cumsum(axis=1).iloc[:, idx-1],
            width = 1,
            color=colors[idx],
            )
    return ax

#### Spin permutation functions ####
def create_bigbrain_spin_permutations(is_downsampled=True, n_perm=1000, batch_size=20):
    """
    Creates spin permutations of the bigbrain surface sphere and stores the
    vertex indices of spins in batches on 'src' folder

    downsample: (bool) use the downsampled (ico5) version instead of ico7
    n_perm: (int) total number of permutations.
    batch_size: (int) number of permutations per batch. Only used if downsample=False
    """
    if is_downsampled:
        outpath = os.path.join(SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz')
        if os.path.exists(outpath):
            print("Spin permutations already exist")
            return
        print(f"Creating {n_perm} spin permutations")
        # read the bigbrain surface sphere files as a mesh that can be used by _generate_spins function
        downsampled_sphere_paths = datasets.load_mesh_paths('sphere', 'bigbrain', downsampled=True)
        lh_sphere = brainspace.mesh.mesh_io.read_surface(downsampled_sphere_paths['L'])
        rh_sphere = brainspace.mesh.mesh_io.read_surface(downsampled_sphere_paths['R'])
        spin_idx = brainspace.null_models.spin._generate_spins(
            lh_sphere, rh_sphere, 
            n_rep = n_perm,
            random_state = 921,
            )
        np.savez_compressed(
            outpath, 
            lh=spin_idx['lh'], 
            rh=spin_idx['rh']
            )
    else:    
        if os.path.exists(os.path.join(SPIN_BATCHES_DIR, 'tpl-bigbrain_desc-spin_indices_batch0.npz')):
            print("Spin permutation batches already exist")
            return
        print(f"Creating {n_perm} spin permutations")
        # read the bigbrain surface sphere files as a mesh that can be used by _generate_spins function
        lh_sphere = brainspace.mesh.mesh_io.read_surface(os.path.join(SRC_DIR, 'tpl-bigbrain_hemi-L_desc-sphere_rot_fsaverage.surf.gii'))
        rh_sphere = brainspace.mesh.mesh_io.read_surface(os.path.join(SRC_DIR, 'tpl-bigbrain_hemi-R_desc-sphere_rot_fsaverage.surf.gii'))
        # create permutations of surface_data with preserved spatial-autocorrelation using _generate_spins
        #  doing it in batches to reduce memory requirements
        n_batch = n_perm // batch_size
        for batch in range(n_batch):
            print(f'\t\tBatch {batch}')
            spin_idx = brainspace.null_models.spin._generate_spins(
                lh_sphere, rh_sphere, 
                n_rep = batch_size,
                random_state = 9*batch, # it's important for random states to be different across batches
                )
            np.savez_compressed(
                os.path.join(
                    SPIN_BATCHES_DIR, f'tpl-bigbrain_desc-spin_indices_batch{batch}.npz'
                    ), 
                lh=spin_idx['lh'], 
                rh=spin_idx['rh']
                )

def spin_test(surface_data_to_spin, surface_data_target, n_perm, is_downsampled):
    """
    Performs spin test on the correlation between `surface_data_to_spin` and 
    `surface_data_target`, where `surface_data_to_spin` is spun

    Parameters
    ----------
    surface_data_to_spin: (np.ndarray | dict of np.ndarray) n_vert * n_features in bigbrain surface space of L and R hemispheres
    surface_data_target: (np.ndarray | dict of np.ndarray) n_vert * n_features in bigbrain surface space of L and R hemispheres
    n_perm: (int) number of spin permutations (max: 1000)
    is_downsampled: (bool) whether the input data is ico5 (downsampled) or ico7
    """
    # TODO: consider removing the option of downsample=False
    # create spins of indices in batches or a single file
    assert n_perm <= 1000
    create_bigbrain_spin_permutations(n_perm=n_perm, is_downsampled=is_downsampled)
    # split hemispheres if the input is concatenated
    if isinstance(surface_data_to_spin, np.ndarray):
        surface_data_to_spin = {
            'L': surface_data_to_spin[:surface_data_to_spin.shape[0]//2],
            'R': surface_data_to_spin[surface_data_to_spin.shape[0]//2:]
        }
    if isinstance(surface_data_target, np.ndarray):
        surface_data_target = {
            'L': surface_data_target[:surface_data_target.shape[0]//2],
            'R': surface_data_target[surface_data_target.shape[0]//2:]
        }
    # calculate test correlation coefficient between all gradients and all other surface maps
    concat_surface_data_to_spin = pd.DataFrame(
        np.concatenate([
            surface_data_to_spin['L'], 
            surface_data_to_spin['R']
            ], axis=0)
        )
    concat_surface_data_target = pd.DataFrame(
        np.concatenate([
            surface_data_target['L'], 
            surface_data_target['R']
            ], axis=0)
        )
    test_r = (
        pd.concat([concat_surface_data_to_spin, concat_surface_data_target], axis=1)
        .corr() # this will calculate the correlation coefficient between all the gradients and other surface maps
        .iloc[:concat_surface_data_to_spin.shape[1], -concat_surface_data_target.shape[1]:] # select only the correlations we are interested in
        .T.values[np.newaxis, :] # convert it to shape (1, n_features_surface_data_target, n_features_surface_data_to_spin)
    )
    null_distribution = test_r.copy() # will have the shape (n_perms, n_features_surface_data_target, n_features_surface_data_to_spin)
    counter = 0
    if is_downsampled:
        batch_files = [os.path.join(SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz')]
    else:
        batch_files = sorted(glob.glob(os.path.join(SPIN_BATCHES_DIR, f'tpl-bigbrain_desc-spin_indices_batch*.npz')))
    for batch_file in batch_files:
        print(f"\t\tBatch {batch_file}")
        # load the batch of spin permutated maps and concatenate left and right hemispheres
        batch_idx = np.load(batch_file) # n_perm * n_vert arrays for 'lh' and 'rh'
        batch_lh_surrogates = surface_data_to_spin['L'][batch_idx['lh']] # n_perm * n_vert * n_features
        batch_rh_surrogates = surface_data_to_spin['R'][batch_idx['rh']]
        concat_batch_surrogates = np.concatenate([batch_lh_surrogates, batch_rh_surrogates], axis=1)
        for perm_idx in range(batch_rh_surrogates.shape[0]):
            if (counter % 100) == 0:
                print(counter)
            surrogate = pd.DataFrame(concat_batch_surrogates[perm_idx, :, :])
            # calculate null correlation coefficient between all gradients and all other surface maps
            null_r = (
                pd.concat([surrogate, concat_surface_data_target], axis=1)
                .corr() # this will calculate the correlation coefficient between all the gradients and other surface maps
                .iloc[:surrogate.shape[1], -concat_surface_data_target.shape[1]:] # select only the correlations we are interested in
                .T.values[np.newaxis, :] # convert it to shape (1, n_features_surface_data_target, n_features_surface_data_to_spin)
            )
            # add this to the null distribution
            null_distribution = np.concatenate([null_distribution, null_r], axis=0)
            counter += 1
            # free up memory
            del surrogate
            gc.collect()
        if counter >= n_perm:
            break
    # remove the test_r from null_distribution
    null_distribution = null_distribution[1:, :, :]
    # calculate p value
    p_val = (np.abs(null_distribution) >= np.abs(test_r)).mean(axis=0)
    # reduce unnecessary dimension of test_r
    test_r = test_r[0, :, :]
    return test_r, p_val, null_distribution

def get_rotated_parcels(parcellation_name, n_perm, excluded_parcels=[], return_indices=True, space='bigbrain'):
    """
    Uses ENIGMA Toolbox approach to spin parcel centroids on the cortical sphere instead
    of spinning all the vertices.
    """
    downsampled = space in ['bigbrain', 'fsaverage']
    rotated_parcels_path = os.path.join(
        SRC_DIR,
        f'tpl-{space}_{"downsampled_" if downsampled else ""}'\
        +f'parc-{parcellation_name}_'\
        +f'excparcs-{len(excluded_parcels)}_'\
        +f'desc-rotated_parcels_n-{n_perm}.npz')
    # TODO: check that excluded parcels are exactly the same
    if os.path.exists(rotated_parcels_path):
        if return_indices:
            return np.load(rotated_parcels_path, allow_pickle=True)['rotated_indices']
        else:
            return np.load(rotated_parcels_path, allow_pickle=True)['rotated_parcels']
    # get the coordinates for the centroids of each parcel
    # on cortical sphere of the space
    centroids = {}
    for hem in ['L', 'R']:
        surf_path = datasets.load_mesh_paths('sphere', space=space, downsampled=downsampled)[hem]
        vertices = nilearn.surface.load_surf_mesh(surf_path).coordinates           
        parc = datasets.load_parcellation_map(parcellation_name, False, downsampled=downsampled, space=space)[hem]
        centroids[hem] = pd.DataFrame(columns=['x','y','z'])
        for parcel_name in np.unique(parc):
            if parcel_name not in (excluded_parcels + MIDLINE_PARCELS.get(parcellation_name, [])):
                this_parc = np.where(parc == parcel_name)[0]
                centroids[hem].loc[parcel_name] = np.mean(vertices[this_parc, :], axis=0)
    # rotate the parcels
    np.random.seed(921)
    rotated_indices = enigmatoolbox.permutation_testing.rotate_parcellation(
        centroids['L'].values.astype('float'),
        centroids['R'].values.astype('float'),
        n_perm
    ).astype('int')
    # get the rotated parcel names
    rotated_parcels = (
        pd.concat([centroids['L'], centroids['R']], axis=0)
        .index.to_numpy()
        [rotated_indices]
    )
    # save/return
    np.savez_compressed(
        rotated_parcels_path, 
        rotated_parcels=rotated_parcels,
        rotated_indices=rotated_indices)
    if return_indices:
        return rotated_indices
    else:
        return rotated_parcels

def spin_test_parcellated(X, Y, parcellation_name, n_perm=1000, space='bigbrain'):
    """
    Uses ENIGMA Toolbox approach to spin parcel centroids on the cortical sphere instead
    of spinning all the vertices. Ideally spin the parcellated data with less NaN values
    by assigining it to X.

    Parameters
    ---------
    X, Y: (pd.DataFrame) n_parcel x n_features
    parcellation_name: (str)
    n_perm: (int)
    surrogates_path: (str | None)
    """
    # calculate test correlation coefficient between all pairs of columns between surface_data_to_spin and surface_data_target
    coefs = (
        pd.concat([X, Y], axis=1)
        # calculate the correlation coefficient between all pairs of columns within and between X and Y
        .corr()
        # select only the correlations we are interested in
        .iloc[:X.shape[1], -Y.shape[1]:] 
        # convert it to shape (1, n_features_Y, n_features_surface_X)
        .T.values[np.newaxis, :] 
    )
    # get the spin rotated parcel indices
    rotated_indices = get_rotated_parcels(parcellation_name, n_perm, return_indices=True, space=space)
    # add NaN parcels back to X and Y so that the number of parcels
    # in them and rotated parcels is the same (to make sure the unlabeled
    # numpy arrays are aligned)
    downsampled = (space in ['bigbrain', 'fsaverage']) 
    X_all_parcels = (parcellate(deparcellate(X, parcellation_name, downsampled=downsampled, space=space), parcellation_name, space=space))
    X_all_parcels = X_all_parcels.loc[~X_all_parcels.index.isin(MIDLINE_PARCELS.get(parcellation_name, []))]
    X_all_parcels.columns = X.columns
    Y_all_parcels = (parcellate(deparcellate(Y, parcellation_name, downsampled=downsampled, space=space), parcellation_name, space=space))
    Y_all_parcels = Y_all_parcels.loc[~Y_all_parcels.index.isin(MIDLINE_PARCELS.get(parcellation_name, []))]
    Y_all_parcels.columns = Y.columns
    assert X_all_parcels.shape[0] == Y_all_parcels.shape[0] == rotated_indices.shape[0]
    null_distribution = np.zeros((n_perm, Y.shape[1], X.shape[1]))
    for x_col in range(X_all_parcels.shape[1]):
        # get all surrogate parcellated maps at once.
        # this involves complicated indexing but the is best way
        # to achieve this extremely more efficiently than loops.
        # if the first permutation in surrogates is e.g. [335, 212, ...]
        # it basically assigns the X 0th value to 335th parcel and
        # its 1st to 212th parcel and so on, and does the same across
        # all the permutations
        surrogates = np.zeros((X_all_parcels.shape[0], n_perm))
        surrogates[
            rotated_indices.T.flatten(),
            np.arange(0, n_perm).reshape(n_perm, 1).repeat(repeats=X_all_parcels.shape[0], axis=1).flatten(),
        ] = np.tile(X_all_parcels.values[:, x_col], n_perm)
        surrogates = pd.DataFrame(surrogates)
        surrogates.columns = [f'surrogate_{i}' for i in range(n_perm)]
        null_distribution[:, :, x_col] = (
            pd.concat([surrogates, Y_all_parcels.reset_index(drop=True)], axis=1)
            .corr()
            .iloc[:surrogates.shape[1], -Y.shape[1]:]
            .values
        )
    # calculate p value
    pvals = (np.abs(null_distribution) > np.abs(coefs)).mean(axis=0)
    # remove unnecessary dimension of test_r
    coefs = coefs[0, :, :]
    pvals = pd.DataFrame(pvals, index=Y.columns, columns=X.columns)
    coefs = pd.DataFrame(coefs, index=Y.columns, columns=X.columns)
    return coefs, pvals, null_distribution

def variogram_test(X, Y, parcellation_name, n_perm=1000, surrogates_path=None, space='bigbrain'):
    """
    Calculates non-parametric p-value of correlation between the columns in X and Y
    by creating surrogates of X with their spatial autocorrelation preserved based
    on variograms. Note that X and Y must be parcellated.
    This is a computationally intensive function and with a single core and for sjh
    parcellation takes about 10 mins for 1000 permutations, if number of features 
    in X and Y is limited to 1

    Parameters
    ----------
    X, Y: (pd.DataFrame) n_parcel x n_features
    parcellation_name: (str)
    n_perm: (int)
    surrogates_path: (str | None)
    """
    if surrogates_path:
        surrogates_path += f'_nperm-{n_perm}_nparcels-{X.shape[0]}.npz'
    # do not recreate the surrogates if they already exists
    # and have the same parcels
    create_surrogates = True
    if surrogates_path and os.path.exists(surrogates_path):
        surrogates = np.load(surrogates_path, allow_pickle=True)['surrogates']
        parcels = np.load(surrogates_path, allow_pickle=True)['parcels']
        if (X.index.values == parcels).all():
            print(f"Surrogates already exist in {surrogates_path} and have the same parcels")
            create_surrogates = False
    if create_surrogates:
        print(f"Creating {n_perm} surrogates based on variograms in {surrogates_path}")
        # load GD
        GD = matrices.DistanceMatrix(parcellation_name, 'geodesic').matrix
        # get parcels (that exist in X) per hemisphere and split the data
        hem_parcels = get_hem_parcels(parcellation_name, limit_to_parcels=X.index.tolist(), space=space)
        GD_hems = {
            'L': GD.loc[hem_parcels['L'], hem_parcels['L']].values,
            'R': GD.loc[hem_parcels['R'], hem_parcels['R']].values
        }
        X_hems = {
            'L': X.loc[hem_parcels['L'], :].values,
            'R': X.loc[hem_parcels['R'], :].values,
        }
        surrogates = {}
        # for gene expression analyses where R hem is empty 
        # skip creating surrogates for it (otherwise it'll throw an error)
        # TODO: write this in a cleaner way
        if len(hem_parcels['R']) == 0:
            hems = ['L']
        else:
            hems = ['L', 'R']
        for hem in hems:
            # load geodesic distance matrices for each hemisphere
            GD_hem = GD_hems[hem]
            # initialize the surrogates
            surrogates[hem] = np.zeros((n_perm, X_hems[hem].shape[0], X_hems[hem].shape[1]))
            for col_idx in range(X_hems[hem].shape[1]):
                # create surrogates
                base = brainsmash.mapgen.base.Base(
                    x = X_hems[hem][:,col_idx], 
                    D = GD_hem,
                    seed=921
                )
                surrogates[hem][:, :, col_idx] = base(n=n_perm)
        # concatenate hemispheres (if R is empty this simply doesn't do anything)
        surrogates = np.concatenate([surrogates[hem] for hem in hems], axis=1) # axis 1 is the parcels
        if surrogates_path:
            np.savez_compressed(surrogates_path, surrogates=surrogates, parcels=X.index.values)
    # calculate test correlation coefficient between all pairs of columns between surface_data_to_spin and surface_data_target
    coefs = (
        pd.concat([X, Y], axis=1)
        # calculate the correlation coefficient between all pairs of columns within and between X and Y
        .corr() 
        # select only the correlations we are interested in
        .iloc[:X.shape[1], -Y.shape[1]:] 
        # convert it to shape (1, n_features_Y, n_features_surface_X)
        .T.values[np.newaxis, :] 
    )
    null_distribution = np.zeros((n_perm, Y.shape[1], X.shape[1]))
    for x_col in range(X.shape[1]):
        # get all surrogate parcellated maps at once.
        # this involves complicated indexing but the is best way
        # to achieve this extremely more efficiently than loops.
        # if the first permutation in surrogates is e.g. [335, 212, ...]
        # it basically assigns the X 0th value to 335th parcel and
        # its 1st to 212th parcel and so on, and does the same across
        # all the permutations
        x_col_surrogates = pd.DataFrame(surrogates[:, :, x_col].T, index=X.index)
        x_col_surrogates.columns = [f'surrogate_{i}' for i in range(n_perm)]
        null_distribution[:, :, x_col] = (
            pd.concat([x_col_surrogates, Y], axis=1)
            .corr()
            .iloc[:x_col_surrogates.shape[1], -Y.shape[1]:]
            .values
        )
    # calculate p value
    pvals = (np.abs(null_distribution) > np.abs(coefs)).mean(axis=0)
    # remove unnecessary dimension of test_r
    coefs = coefs[0, :, :]
    pvals = pd.DataFrame(pvals, index=Y.columns, columns=X.columns)
    coefs = pd.DataFrame(coefs, index=Y.columns, columns=X.columns)
    return coefs, pvals, null_distribution

def exponential_eval(x, a, b, c):
    """
    Evaluates y given x and exponential fit parameters
    in form y = a + b * exp(c*x)
    """
    return a + b * np.exp(c * x)

def exponential_fit(x, y):
    """
    Computes an exponential decay fit to two vectors of x and y data
    result is in form y = a + b * exp(c*x).
    It first calculates an analytical approximation based on 
    https://math.stackexchange.com/questions/2318418/initial-guess-for-fitting-exponential-with-offset
    (which is originally based on "Regressions et Equations integrales" by Jean Jacquelin 
    [https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales])
    and is converted to Python by https://gist.github.com/johanvdw/443a820a7f4ffa7e9f8997481d7ca8b3.
    Then uses the analytical solution as the primer to scipy.optimize.curve_fit
    """
    # TODO: add comments to the code copied from the gist
    #>>> start of the gist
    n = np.size(x)
    # sort the data into ascending x order
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    Sk = np.zeros(n)
    for n in range(1,n):
        Sk[n] = Sk[n-1] + (y[n] + y[n-1])*(x[n]-x[n-1])/2
    dx = x - x[0]
    dy = y - y[0]
    m1 = np.matrix([[np.sum(dx**2), np.sum(dx*Sk)],
                    [np.sum(dx*Sk), np.sum(Sk**2)]])
    m2 = np.matrix([np.sum(dx*dy), np.sum(dy*Sk)])
    [d, c] = (m1.I * m2.T).flat
    m3 = np.matrix([[n,                  np.sum(np.exp(  c*x))],
                    [np.sum(np.exp(c*x)),np.sum(np.exp(2*c*x))]])
    m4 = np.matrix([np.sum(y), np.sum(y*np.exp(c*x).T)])
    [a, b] = (m3.I * m4.T).flat
    coefs  = [a, b, c]
    #<<< end of the gist
    # try improving the analytical coefs
    try:
        coefs, _ = scipy.optimize.curve_fit(exponential_eval, x, y, p0=coefs, maxfev=10000)
    except RuntimeError: 
        # if it failed to convergence after 10000 iterations
        # use the original analytical solution
        print("curve_fit failed to converge after 10000 iterations")
        pass
    return coefs

###### Transforms #########

def fsa_annot_to_fsa5_gii(parcellation_name):
    """
    Converts a parcellation from fsaverage space to fsaverage5
    simply by taking first 10242 vertices and saving it again
    as .annot and .label.gii files
    It is needed for abagen 
    """
    for hem in ['lh', 'rh']:
        # load parcellation map, ctab and names of the original annot file
        ico7_annot_path = os.path.join(SRC_DIR, f'{hem}_{parcellation_name}.annot')
        labels_ico7, ctab, names = nibabel.freesurfer.io.read_annot(ico7_annot_path)
        # select the first 10k vertices of ico7 to get ico5 parcellation map
        labels_ico5 = labels_ico7[:datasets.N_VERTICES_HEM_BB_ICO5]
        if parcellation_name == 'aparc':
            # the background in aparc has the value -1
            # but abagen assumes in every parcellation for
            # 0 to be the value for the background
            labels_ico5[labels_ico5==-1] = 0
        # save as annot
        nibabel.freesurfer.io.write_annot(
            ico7_annot_path.replace('.annot', '_fsa5.annot'),
            labels_ico5, ctab, names
        )
        # convert to gifti and save it
        gifti_img = abagen.images.annot_to_gifti(ico7_annot_path.replace('.annot', '_fsa5.annot'))
        nibabel.save(
            gifti_img,
            ico7_annot_path.replace('.annot', '_fsa5.label.gii')
            )

def write_gifti(out_path, points=None, faces=None, data=None):
    """
    Write gifti mesh and/or data.

    Parameters
    ---------
    out_path: (str)
    points: (np.ndarrray)
    faces: (np.ndarray)
    data: (np.ndarray)
    
    Credit: adapeted from https://github.com/nighres/nighres/blob/master/nighres/io/io_mesh.py
    """
    
    arrays = []
    if points is not None:
        coord_array = nibabel.gifti.GiftiDataArray(data=points,
                                                intent=nibabel.nifti1.intent_codes[
                                                'NIFTI_INTENT_POINTSET'],
                                                datatype='NIFTI_TYPE_FLOAT32')
        face_array = nibabel.gifti.GiftiDataArray(data=faces,
                                            intent=nibabel.nifti1.intent_codes[
                                                'NIFTI_INTENT_TRIANGLE'],
                                                datatype='NIFTI_TYPE_FLOAT32')
        arrays += [coord_array, face_array]
    if data is not None:
        data_array = nibabel.gifti.GiftiDataArray(data=data,
                                         intent=nibabel.nifti1.intent_codes[
                                             'NIFTI_INTENT_ESTIMATE'],
                                             datatype='NIFTI_TYPE_FLOAT32')
        arrays += [data_array]
    gii = nibabel.gifti.GiftiImage(darrays=arrays)
    nibabel.save(gii, out_path)

def surface_to_surface_transform(
        in_dir, out_dir, in_lh, in_rh, 
        in_space, out_space, desc, interp='nearest',
        bbwarp_singularity_path = '/data/group/cng/Tools/bigbrainwarp.simg'):
    """
    Wrapper for running bigbrainwarp singularity from python
    """
    if not os.path.exists(bbwarp_singularity_path):
        print("Singularity image not found")
        return
    subprocess.call(f"""
        singularity exec --cleanenv {bbwarp_singularity_path} \
        /bin/bash -c \
        "source /BigBrainWarp/scripts/init.sh && \
        cd {in_dir} && \
        /BigBrainWarp/bigbrainwarp \
        --in_lh '{os.path.join(in_dir, in_lh)}' \
        --in_rh '{os.path.join(in_dir, in_rh)}' \
        --in_space {in_space} \
        --out_space {out_space} \
        --interp {interp} \
        --desc {desc} \
        --wd {out_dir}"
    """, shell=True)

def macaque_to_human(in_paths, desc):
    """
    Wrapper for wb_command to transform from macaque to human
    cortex (yerkes 32k space to fs_LR 32k space)

    Parameters
    --------
    in_paths: (dict)
        path to L and R hemi .gii files
    desc: (str)

    Returns
    -------
    out_paths: (dict)
        path to L and R hemi .gii files
    
    Credit: based on Xu 2020 Neuroimage approach https://github.com/TingsterX/alignment_macaque-human
    """
    dir_path = os.path.dirname(in_paths['L'])
    out_paths = {}
    for hem in ['L', 'R']:
        out_paths[hem] = os.path.join(dir_path, f'tpl-fs_LR_hemi-{hem}_den-32k_desc-{desc}.shape.gii')
        subprocess.call(f"""
            wb_command -metric-resample \
            {in_paths['L']} \
            {os.path.join(SRC_DIR, f'{hem}.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii')} \
            {os.path.join(SRC_DIR, f'S1200.{hem}.sphere.32k_fs_LR.surf.gii')} \
            'BARYCENTRIC' \
            {out_paths[hem]}
        """, shell=True)
    return out_paths

def human_to_macaque(in_paths, desc):
    """
    Wrapper for wb_command to transform from human to macaque
    cortex (fs_LR 32k space to yerkes 32k space)

    Parameters
    --------
    in_paths: (dict)
        path to L and R hemi .gii files
    desc: (str)

    Returns
    -------
    out_paths: (dict)
        path to L and R hemi .gii files

    Credit: based on Xu 2020 Neuroimage approach https://github.com/TingsterX/alignment_macaque-human
    """
    dir_path = os.path.dirname(in_paths['L'])
    out_paths = {}
    for hem in ['L', 'R']:
        out_paths[hem] = os.path.join(dir_path, f'tpl-yerkes_hemi-{hem}_den-32k_desc-{desc}.shape.gii')
        subprocess.call(f"""
            wb_command -metric-resample \
            {in_paths['L']} \
            {os.path.join(SRC_DIR, f'{hem}.human-to-macaque.sphere.reg.32k_fs_LR.surf.gii')} \
            {os.path.join(SRC_DIR, f'MacaqueYerkes19.{hem}.sphere.32k_fs_LR.surf.gii')} \
            'BARYCENTRIC' \
            {out_paths[hem]}
        """, shell=True)
    return out_paths