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
import nilearn.surface
import nilearn.plotting
import nibabel
import statsmodels.api as sm
import scipy.io
import abagen

import datasets
import matrices

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
SPIN_BATCHES_DIR = os.path.join(SRC_DIR, 'spin_batches')
os.makedirs(SPIN_BATCHES_DIR, exist_ok=True)

MIDLINE_PARCELS = {
    'schaefer400': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'schaefer1000': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'sjh': [0],
    'aparc': ['L_unknown', 'None'],
    'mmp1': ['???'],
    'brodmann': ['???']
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
    #> Connect
    ftp = FTP('bigbrain.loris.ca')
    ftp.login()
    #> Go to dir
    ftp.cwd(ftp_dir)
    #> Determine filename
    if not out_filename:
        out_filename = ftp_filename
    print('Downloading', out_filename, end=' ')
    #> Download
    if not os.path.exists(out_filename):
        with open(out_filename, 'wb') as file_obj:
            ftp.retrbinary(f'RETR {ftp_filename}', file_obj.write)
        print(">> done")
    else:
        print(">> already exists")
    #> Copy to another folder if needed
    if copy_to:
        if not os.path.exists(copy_to):
            shutil.copyfile(out_filename, copy_to)

###### Data manipulation ######

def parcellate(surface_data, parcellation_name, averaging_method='median', 
               na_midline=True):
    """
    Parcellates `surface data` using `parcellation` and by taking the
    median or mean (specified via `averaging_method`) of the vertices within each parcel.

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) n_vertices x n_features surface data of L and R hemispheres
    parcellation_name: (str) Parcellation scheme
    averaging_method: (str) Method of averaging over vertices within a parcel. Default: 'median'
        - 'median'
        - 'mean'
        - None (will return groupby object)
    na_midline: (bool) make midline vertices NaN

    Returns
    ---------
    parcellated_data: (pd.DataFrame or dict of pd.DataFrame) n_parcels x n_features for data of L and R hemispheres or both hemispheres
    """
    if isinstance(surface_data, dict):
        #> determine if the surface data is downsampled
        is_downsampled = (surface_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB_ICO5)
        #> load parcellation map
        labeled_parcellation_maps = datasets.load_parcellation_map(
            parcellation_name, 
            concatenate=False,
            downsampled=is_downsampled)
        parcellated_data = {}
        for hem in ['L', 'R']:
            #> parcellate
            parcellated_vertices = (
                pd.DataFrame(surface_data[hem], index=labeled_parcellation_maps[hem])
            )
            #> remove midline data
            if na_midline:
                for midline_parcel in MIDLINE_PARCELS[parcellation_name]:
                    if midline_parcel in parcellated_vertices.index:
                        parcellated_vertices.loc[midline_parcel] = np.NaN
            parcellated_vertices = (parcellated_vertices
                .reset_index(drop=False)
                .groupby('index')
            )
            #> operate on groupby object if needed
            if averaging_method == 'median':
                parcellated_data[hem] = parcellated_vertices.median()
            elif averaging_method == 'mean':
                parcellated_data[hem] = parcellated_vertices.mean()
            else:
                parcellated_data[hem] = parcellated_vertices
    elif isinstance(surface_data, np.ndarray):
        #> determine if the surface data is downsampled
        is_downsampled = (surface_data.shape[0] == datasets.N_VERTICES_HEM_BB_ICO5*2)
        #> load parcellation map
        labeled_parcellation_maps = datasets.load_parcellation_map(
            parcellation_name, 
            concatenate=True,
            downsampled=is_downsampled)
        #> parcellate
        parcellated_vertices = (
            pd.DataFrame(surface_data, index=labeled_parcellation_maps)
        )
        if na_midline:
            for midline_parcel in MIDLINE_PARCELS[parcellation_name]:
                if midline_parcel in parcellated_vertices.index:
                    parcellated_vertices.loc[midline_parcel] = np.NaN
        parcellated_vertices = (parcellated_vertices
            .reset_index(drop=False)
            .groupby('index')
        )
        #> operate on groupby object if needed
        if averaging_method == 'median':
            parcellated_data = parcellated_vertices.median()
        elif averaging_method == 'mean':
            parcellated_data = parcellated_vertices.mean()
        else:
            parcellated_data = parcellated_vertices
    return parcellated_data

def concat_hemispheres(parcellated_data, dropna=False):
    """
    Concatenates the parcellated data of L and R hemispheres

    Parameters
    ----------
    parcellated_data: (dict of pd.DataFrame) n_parcels x 6 for laminar data of L and R hemispheres
    dropna: (bool) remove parcels with no data. Default: False
    TODO: merge this with parcellate

    Returns
    ----------
    concat_data: (pd.DataFrame) n_parcels*2 x 6
    """
    #> concatenate hemispheres and drop NaN if needed
    concat_data = (pd.concat(
        [
            parcellated_data['L'], 
            parcellated_data['R']
        ],
        axis=0)
        .reset_index(drop=False) # take the average values in case parcels are repeated in L and R
        .groupby('index').mean()
        )
    if dropna:
        concat_data = concat_data.dropna()
    return concat_data

def downsample(surface_data):
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

    Returns
    ---------
    downsampled_surface_data: (np.ndarray or dict of np.ndarray) n_vertices_ico5 x n_features
    """
    concat_output = False
    if isinstance(surface_data, np.ndarray):
        surface_data = {
            'L': surface_data[:datasets.N_VERTICES_HEM_BB, :],
            'R': surface_data[datasets.N_VERTICES_HEM_BB:, :],
        }
        concat_output = True
    downsampled_surface_data = {}
    for hem in ['L', 'R']:
        #> load the downsampled surface from matlab results
        mat = scipy.io.loadmat(
            os.path.join(
                SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
        )
        #> load the "parcellation map" of downsampled bigbrain surface
        #  indicating the index for the center vertex of each parcel
        #  in the original ico7 (320k) BB surface
        downsample_parcellation = mat['nn_bb'][0, :]-1 #1-indexing to 0-indexing
        #> then "parcellate" the ico7 data to ico5 by taking
        #  the average of center vertices and their neighbors
        #  indicated in 'nn_bb'
        downsampled = (
            pd.DataFrame(surface_data[hem], index=downsample_parcellation)
            .reset_index(drop=False)
            .groupby('index')
            .mean()
        )
        #> load correctly ordered indices of center vertices
        #  and reorder downsampled data to the correct order alligned with ico5 surface
        bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
        downsampled_surface_data[hem] = downsampled.loc[bb_downsample_indices].values
    if concat_output:
        return np.concatenate([
            downsampled_surface_data['L'],
            downsampled_surface_data['R']
            ], axis=0)
    else:
        return downsampled_surface_data


def regress_out_matrix_covariates(input_matrix, cov_matrices, pos_only=True):
    """
    Fits `cov_matrices` to `input_matrix` and return the residual. Before
    the fitting the lower triangle of matrices is flattened and depending on
    `pos_only` zero and negative values are removed

    Parameters
    ----------
    input_matrix: (np.ndarray) parc x parc laminar similarity matrix
    cov_matrices: (list of np.ndarray) parc x parc covariate matrix
    pos_only: (bool) only include positive values of input_matrix

    Return
    ---------
    cleaned_matrix: (np.ndarray) parc x parc cleaned laminar similarity matrix
    """
    #> read y: only keep lower triangle (ignoring diagonal) and reshape to (n_parc_pair, 1)
    tril_indices = np.tril_indices_from(input_matrix, -1)
    y = input_matrix[tril_indices].reshape(-1, 1)
    #> create X: add intercept
    X = np.onses_like(y)
    #> add lower triangle of cov_matrices to X
    for cov_matrix in cov_matrices:
        X = np.hstack(
            X,
            cov_matrix[tril_indices].reshape(-1, 1)
        )
    #> create the mask of positive values
    if pos_only:
        mask = (y > 0)
    else:
        mask = np.array([True] * y.shape[0])
    #> regression
    resid = sm.OLS(y[mask, :], X[mask, :]).fit().resid
    #> project back resid to the input_matrix shape
    # lower triangle
    cleaned_tril = np.zeros(y.shape[1])
    cleaned_tril[mask] = resid
    cleaned_matrix = np.zeros_like(input_matrix)
    cleaned_matrix[tril_indices] = cleaned_tril
    # upper triagnle
    triu_indices = np.triu_indices_from(input_matrix, 1) #ignoring diagonal
    cleaned_matrix[triu_indices] = cleaned_matrix.T[triu_indices]
    # diagonal = 1 (since the input is similarity)
    cleaned_matrix[np.diag_indices_from(cleaned_matrix)] = 1
    return cleaned_matrix

def regress_out_surf_covariates(input_surface_data, cov_surface_data, sig_only=False, renormalize=False):
    """
    Fits `cov_surface_data` to `input_surface_data` and return the residual.

    Parameters
    ----------
    input_surface_data: (np.ndarray) n_vertices x n_cols input surface data
    cov_surface_data: (np.ndarray) n_vertices covariate surface data
                     TODO: add support for multiple covariates
    sig_only: (bool) do the regression only if the correlation is significant
                     TODO: for this also consider doing spin permutation
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
        print("col", col_idx)
        #> read and reshape y
        y = input_surface_data[mask, col_idx].reshape(-1, 1)
        #> demean and store mean (instead of fitting intercept)
        y_mean = y.mean()
        y_demean = y - y_mean
        #> convert covariate shape to n_vert * 1 if there's only one
        X = cov_surface_data.reshape(-1, 1)
        X = X[mask, :]
        assert X.shape[0] == y.shape[0]
        #> model fitting
        lr = sm.OLS(y_demean, X).fit()
        print("pval", lr.pvalues[0], "*" if lr.pvalues[0] < 0.05 else "")
        if (lr.pvalues[0] < 0.05) or (not sig_only):
            cleaned_surface_data[mask, col_idx] = lr.resid + y_mean
        else:
            cleaned_surface_data[mask, col_idx] = y[:, 0]
    if renormalize:
        cleaned_surface_data /= cleaned_surface_data.sum(axis=1, keepdims=True)    
    return cleaned_surface_data

def deparcellate(parcellated_data, parcellation_name, downsampled=False):
    """
    Project the parcellated data to surface vertices while handling empty parcels
    (parcels that are not in the parcellated data but are in the parcellation map)

    Parameters
    ----------
    parcellated_data: (pd.DataFrame | pd.Series) n_parcels x n_features
    parcellation_name: (str)
    downsampled: (bool)

    Returns
    -------
    surface_map: (np.ndarray) n_vertices [both hemispheres] x n_features
    """
    #> load concatenated parcellation map
    concat_parcellation_map = datasets.load_parcellation_map(
        parcellation_name, concatenate=True, downsampled=downsampled)
    #> load dummy parcellated data covering the whole brain
    if downsampled:
        dummy_surf_data = np.zeros(datasets.N_VERTICES_HEM_BB_ICO5)
    else:
        dummy_surf_data = np.zeros(datasets.N_VERTICES_HEM_BB)
    parcellated_dummy = parcellate(
        {'L': dummy_surf_data,
         'R': dummy_surf_data,},
         parcellation_name)
    concat_parcellated_dummy = concat_hemispheres(parcellated_dummy, dropna=False)
    all_parcels = concat_parcellated_dummy.index.to_series().rename('parcel')
    #> create a dataframe including all parcels, where invalid parcels are NaN
    #   (this is necessary to be able to project it to the parcellation)
    labeled_parcellated_data = pd.concat(
        [
            parcellated_data,
            all_parcels
        ], axis=1).set_index('parcel')
    #> get the surface map by indexing the parcellated map at parcellation labels
    surface_map = labeled_parcellated_data.loc[concat_parcellation_map].values # shape: vertex X gradient
    # TODO: convert it back to DataFrame or Series with original col names
    return surface_map

def get_split_hem_idx(parcellation_name, exc_regions):
    """
    Get the index of the first RH parcel to split hemispheres

    Parameters
    ----------
    parcellation_name: (str)
    exc_regions: (str or None)
        - adysgranular
        - allocortex
        - None
    """
    if exc_regions is not None:
        exc_masks = datasets.load_exc_masks(exc_regions, parcellation_name)
    else:
        exc_masks = {hem:np.zeros(datasets.N_VERTICES_HEM_BB) for hem in ['L', 'R']}
    parcels_to_exclude = parcellate(exc_masks, parcellation_name)
    split_hem_idx = int(np.nansum(1-parcels_to_exclude['L'].values)) # number of non-exc-mask parcels in lh
    return split_hem_idx

def get_parcel_center_indices(parcellation_name):
    """
    Gets the center of parcels and returns their index
    in BigBrain ico7 surface

    Based on "geoDistMapper.py" from micapipe/functions
    Original Credit:
    # Translated from matlab:
    # Original script by Boris Bernhardt and modified by Casey Paquola
    # Translated to python by Jessica Royer
    """
    centers = {}
    for hem in ['L', 'R']:
        out_path = os.path.join(
            SRC_DIR,
            f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation_centers.csv'
            )
        if os.path.exists(out_path):
            centers[hem] = pd.read_csv(out_path, index_col='parcel').iloc[:, 0]
            continue
        #> load surf
        surf_path = os.path.join(
            SRC_DIR, 
            f'tpl-bigbrain_hemi-L_desc-mid.surf.gii'
            )
        vertices = nilearn.surface.load_surf_mesh(surf_path).coordinates           
        parc = datasets.load_parcellation_map(parcellation_name, False)[hem]

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

def get_neighbors_mask(parcellation_name, proportion=0.2, exc_regions='adysgranular'):
    """
    Returns a matrix with each row indicating neighbors to any given parcel,
    which are among the `proportion` closest parcel to the seed parcel in
    the same hemisphere. Neighbors also include the seed. This can then be
    used in spatial leave one-out cross-validation

    Parameters
    ----------
    parcellation_name: (str)
    proportion: (float)
        proportion of parcels assigned to the train set for each parcel.
    exc_regions: (str | None) 
        exclude adysgranular regions
    Returns
    -------
    neighbors: (pd.DataFrame) 
        n_parc x n_parc with each row representing the neighbors
        of a given parcel
    """
    gd = matrices.DistanceMatrix(
        parcellation_name=parcellation_name, 
        kind='geodesic', 
        exc_regions=exc_regions)
    neighbors = (
        gd.matrix
        #> remove seed and contralateral parcels
        .replace(0, np.NaN)
        #> select 20% closest parcels (in the same hemisphere)
        .apply(lambda row: row < row.quantile(proportion), axis=1)
        #> add the seed parcel to the neighbors
        + np.eye(gd.matrix.shape[0]) 
        #> convert to bool
        .astype('bool')
    )
    return neighbors

###### Plotting ######
def make_colorbar(vmin, vmax, cmap=None, bins=None, orientation='vertical', figsize=None):
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

def plot_matrix(matrix, outpath, cmap="rocket", vrange=(0.025, 0.975), **kwargs):
    """
    Plot the matrix as heatmap

    Parameters
    ----------
    matrix: (np.ndarray) n x k*n square matrix or horizontally concatenated
        square matrices
    cmap: (str or colormap objects) colormap recognizable by seaborn
    vrange: (tuple of size 2) vmin and vmax as percentiles (for whole range put (0, 1))
    """
    n_square_matrices = matrix.shape[1] // matrix.shape[0]
    vmin = np.quantile(matrix.flatten(),vrange[0])
    vmax = np.quantile(matrix.flatten(),vrange[1])
    fig, ax = plt.subplots(figsize=(7 * n_square_matrices,7))
    sns.heatmap(
        matrix,
        vmin=vmin, vmax=vmax,
        cmap=cmap, cbar=False,
        ax=ax)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=192)
    clbar_fig = make_colorbar(
        vmin=vmin, vmax=vmax,
        cmap=cmap, orientation='vertical'
        )
    clbar_fig.savefig(outpath+'_clbar', dpi=192)

def plot_surface(surface_data, filename=None, space='bigbrain', inflate=False, 
                 plot_downsampled=True, layout_style='horizontal', cmap='viridis',
                 toolbox='brainspace'):
    """
    Plots the surface data with medial and lateral views of both hemispheres

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) (n_vert,) 
    filename: (str | None) 
        path to output without .png
    space: (str)
        - bigbrain
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
    """
    #> split surface if it has been concatenated (e.g. gradients)
    #  and make sure the shape is correct
    if isinstance(surface_data, np.ndarray):
        assert surface_data.shape[0] == datasets.N_VERTICES_HEM_BB * 2
        lh_surface_data = surface_data[:datasets.N_VERTICES_HEM_BB]
        rh_surface_data = surface_data[datasets.N_VERTICES_HEM_BB:]
        surface_data = {'L': lh_surface_data, 'R': rh_surface_data}
    else:
        assert surface_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB
    #> specify the mesh and downsample the data if needed
    if plot_downsampled:
        if (surface_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB):
            surface_data = downsample(surface_data)
        if inflate:
            mesh_paths = datasets.load_downsampled_surface_paths('inflated')
        else:
            mesh_paths = datasets.load_downsampled_surface_paths('orig')
    else:
        if inflate:
            mesh_paths = {
                'L': os.path.join(
                    SRC_DIR, 'tpl-bigbrain_hemi-L_desc-mid.surf.inflate.gii'
                    ),
                'R': os.path.join(
                    SRC_DIR, 'tpl-bigbrain_hemi-R_desc-mid.surf.inflate.gii'
                    )
            }
        else:
            mesh_paths = {
                'L': os.path.join(
                    SRC_DIR, 'tpl-bigbrain_hemi-L_desc-mid.surf.gii'
                    ),
                'R': os.path.join(
                    SRC_DIR, 'tpl-bigbrain_hemi-R_desc-mid.surf.gii'
                    )
            }
    if toolbox == 'brainspace':
        _plot_brainspace(surface_data, mesh_paths, filename, layout_style, cmap)
    else:
        _plot_nilearn(surface_data, mesh_paths, filename, layout_style, cmap)
    # TODO plot colorbar separately

def _plot_brainspace(surface_data, mesh_paths, filename, layout_style, cmap):
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
    #> rejoin the hemispheres
    surface_data = np.concatenate([surface_data['L'], surface_data['R']]).flatten()
    #> load bigbrain surfaces
    lh_surf = brainspace.mesh.mesh_io.read_surface(mesh_paths['L'])
    rh_surf = brainspace.mesh.mesh_io.read_surface(mesh_paths['R'])
    #> read surface data files and concatenate L and R
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
        cmap = cmap, color_bar=False,
        # TODO: change size and zoom based on layout 
        size=size, zoom=zoom,
        interactive=False, embed_nb=embed_nb,
        screenshot=screenshot, filename=filename, 
        transparent_bg=True)

def _plot_nilearn(surface_data, mesh_paths, filename, layout_style, cmap):
    """
    Plots `surface_data` on `mesh_paths` using nilearn
    """
    #> initialize the figures
    if layout_style == 'horizontal':
        figure, axes = plt.subplots(1, 4, figsize=(24, 5), subplot_kw={'projection': '3d'})
    elif layout_style == 'grid':
        figure, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        #> reorder axes so that lateral views are on top, matching the order of axes
        #  in the horizontal layout
        axes = np.array([axes[0, 0], axes[1, 0], axes[1, 1], axes[0, 1]])
    curr_ax_idx = 0
    for hemi in ['left', 'right']:
        mesh_path = mesh_paths[hemi[0].upper()]
        #> specify the view order
        if hemi == 'left':
            views_order = ['lateral', 'medial']
        else:
            views_order = ['medial', 'lateral']
        #> plot
        for view in views_order:
            nilearn.plotting.plot_surf(
                mesh_path,
                surface_data[hemi[0].upper()],
                hemi=hemi, view=view, axes=axes[curr_ax_idx],
                cmap=cmap,
                )
            curr_ax_idx += 1
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    if filename:
        figure.savefig(filename, dpi=192)
    else:
        return figure


#### Spin permutation functions ####
def create_bigbrain_spin_permutations(is_downsampled=True, n_perm=1000, batch_size=20):
    """
    Creates spin permutations of the bigbrain surface sphere and stores the
    vertex indices of spins in batches on 'src' folder

    downsample: (bool) use the downsampled (ico5) version instead of ico7
    n_perm: (int) total number of permutations.
    batch_size: (int) number of permutations per batch. Only used if downsample=False
    """
    #TODO clean the code
    if is_downsampled:
        outpath = os.path.join(SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz')
        if os.path.exists(outpath):
            print("Spin permutations already exist")
            return
        print(f"Creating {n_perm} spin permutations")
        #> read the bigbrain surface sphere files as a mesh that can be used by _generate_spins function
        downsampled_sphere_paths = datasets.load_downsampled_surface_paths('sphere')
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
        #> read the bigbrain surface sphere files as a mesh that can be used by _generate_spins function
        lh_sphere = brainspace.mesh.mesh_io.read_surface(os.path.join(SRC_DIR, 'tpl-bigbrain_hemi-L_desc-sphere_rot_fsaverage.surf.gii'))
        rh_sphere = brainspace.mesh.mesh_io.read_surface(os.path.join(SRC_DIR, 'tpl-bigbrain_hemi-R_desc-sphere_rot_fsaverage.surf.gii'))
        #> create permutations of surface_data with preserved spatial-autocorrelation using _generate_spins
        #  doing it in batches to reduce memory requirements
        n_batch = n_perm // batch_size
        for batch in range(n_batch):
            print('\t\tBatch', batch)
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
    #> create spins of indices in batches or a single file
    assert n_perm <= 1000
    create_bigbrain_spin_permutations(n_perm=n_perm, is_downsampled=is_downsampled)
    #> split hemispheres if the input is concatenated
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
    #> calculate test correlation coefficient between all gradients and all other surface maps
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
        print("\t\tBatch", batch_file)
        #> load the batch of spin permutated maps and concatenate left and right hemispheres
        batch_idx = np.load(batch_file) # n_perm * n_vert arrays for 'lh' and 'rh'
        batch_lh_surrogates = surface_data_to_spin['L'][batch_idx['lh']] # n_perm * n_vert * n_features
        batch_rh_surrogates = surface_data_to_spin['R'][batch_idx['rh']]
        concat_batch_surrogates = np.concatenate([batch_lh_surrogates, batch_rh_surrogates], axis=1)
        for perm_idx in range(batch_rh_surrogates.shape[0]):
            print(counter)
            surrogate = pd.DataFrame(concat_batch_surrogates[perm_idx, :, :])
            #> calculate null correlation coefficient between all gradients and all other surface maps
            null_r = (
                pd.concat([surrogate, concat_surface_data_target], axis=1)
                .corr() # this will calculate the correlation coefficient between all the gradients and other surface maps
                .iloc[:surrogate.shape[1], -concat_surface_data_target.shape[1]:] # select only the correlations we are interested in
                .T.values[np.newaxis, :] # convert it to shape (1, n_features_surface_data_target, n_features_surface_data_to_spin)
            )
            #> add this to the null distribution
            null_distribution = np.concatenate([null_distribution, null_r], axis=0)
            counter += 1
            #> free up memory
            del surrogate
            gc.collect()
        if counter >= n_perm:
            break
    #> remove the test_r from null_distribution
    null_distribution = null_distribution[1:, :, :]
    #> calculate p value
    p_val = (np.abs(null_distribution) >= np.abs(test_r)).mean(axis=0)
    #> reduce unnecessary dimension of test_r
    test_r = test_r[0, :, :]
    return test_r, p_val, null_distribution

def variogram_test(X, Y, parcellation_name, exc_regions, n_perm=1000, surrogates_path=None):
    """
    Calculates non-parametric p-value of correlation between the columns in X and Y
    by creating surrogates of X with their spatial autocorrelation preserved based
    on variograms. Note that X and Y must be parcellated.

    Parameters
    ----------
    X, Y: (pd.DataFrame) n_parcel x n_features
    parcellation_name: (str)
    exc_regions: (str)
    n_perm: (int)
    surrogates_path: (str | None)
    """
    if surrogates_path:
        surrogates_path += f'_nperm-{n_perm}.npz'
    if surrogates_path and os.path.exists(surrogates_path):
        print("Surrogates already exist in", surrogates_path)
        surrogates = np.load(surrogates_path)['surrogates']
    else:
        print(f"Creating {n_perm} surrogates based on variograms in {surrogates_path}")
        GD = matrices.DistanceMatrix(parcellation_name, 'geodesic', exc_regions=exc_regions).matrix.values
        split_hem_idx = get_split_hem_idx(parcellation_name, exc_regions)
        GD_hems = {
            'L': GD[:split_hem_idx, :split_hem_idx],
            'R': GD[split_hem_idx:, split_hem_idx:]
        }
        X_hems = {
            'L': X.values[:split_hem_idx, :],
            'R': X.values[split_hem_idx:, :],
        }
        surrogates = {}
        for hem in ['L', 'R']:
            #> load geodesic distance matrices for each hemisphere
            GD_hem = GD_hems[hem]
            #> initialize the surrogates
            surrogates[hem] = np.zeros((n_perm, X_hems[hem].shape[0], X_hems[hem].shape[1]))
            for col_idx in range(X_hems[hem].shape[1]):
                #> create surrogates
                base = brainsmash.mapgen.base.Base(
                    x = X_hems[hem][:,col_idx], 
                    D = GD_hem,
                    seed=921 # TODO: is it okay to have a fixed seed?
                )
                surrogates[hem][:, :, col_idx] = base(n=n_perm)
        #> concatenate hemispheres
        surrogates = np.concatenate([
            surrogates['L'], 
            surrogates['R'] 
        ], axis=1) # axis 1 is the parcels
        if surrogates_path:
            np.savez_compressed(surrogates_path, surrogates=surrogates)
    #> calculate test correlation coefficient between all pairs of columns between surface_data_to_spin and surface_data_target
    test_r = (
        pd.concat([X, Y], axis=1)
        # calculate the correlation coefficient between all pairs of columns within and between X and Y
        .corr() 
        # select only the correlations we are interested in
        .iloc[:X.shape[1], -Y.shape[1]:] 
        # convert it to shape (1, n_features_Y, n_features_surface_X)
        .T.values[np.newaxis, :] 
    )
    #> keep track of null distribution of correlation coefficients
    null_distribution = test_r.copy() # will have the shape (n_perms, n_features_surface_data_target, n_features_surface_data_to_spin)
    for surrogate_idx in range(n_perm):
        curr_surrogate = pd.DataFrame(surrogates[surrogate_idx, :, :], index=X.index)
        #> calculate null correlation coefficient between all pairs of columns between surface_data_to_spin and surface_data_target
        null_r = (
            pd.concat([curr_surrogate, Y], axis=1)
            .corr()
            .iloc[:curr_surrogate.shape[1], -Y.shape[1]:]
            .T.values[np.newaxis, :]
        )
        #> add this to the null distribution
        null_distribution = np.concatenate([null_distribution, null_r], axis=0)
        #> free up memory
        gc.collect()
    #> remove the test_r from null_distribution
    null_distribution = null_distribution[1:, :, :]
    #> calculate p value
    p_val = (np.abs(null_distribution) > np.abs(test_r)).mean(axis=0)
    #> remove unnecessary dimension of test_r
    test_r = test_r[0, :, :]
    return test_r, p_val, null_distribution

def fsa_annot_to_fsa5_gii(parcellation_name):
    """
    Converts a parcellation from fsaverage space to fsaverage5
    simply by taking first 10242 vertices and saving it again
    as .annot and .label.gii files
    """
    for hem in ['lh', 'rh']:
        ico7_annot_path = os.path.join(SRC_DIR, f'{hem}_{parcellation_name}.annot')
        labels_ico7, ctab, names = nibabel.freesurfer.io.read_annot(ico7_annot_path)
        labels_ico5 = labels_ico7[:datasets.N_VERTICES_HEM_BB_ICO5]
        nibabel.freesurfer.io.write_annot(
            ico7_annot_path.replace('.annot', '_fsa5.annot'),
            labels_ico5, ctab, names
        )
        gifti_img = abagen.images.annot_to_gifti(ico7_annot_path.replace('.annot', '_fsa5.annot'))
        nibabel.save(
            gifti_img,
            ico7_annot_path.replace('.annot', '_fsa5.label.gii')
            )

def fix_brodmann_annot():
    """
    The original .annot file from FreeSurfer for Brodmann
    includes non-brodmann regions as well. This function 
    fixes this issue by replacing non-Brodmann parcels with
    '???' (background)
    """
    for hem in ['lh', 'rh']:
        annot_path = os.path.join(SRC_DIR, f'{hem}_brodmann_orig.annot')
        labels, ctab, names = nibabel.freesurfer.io.read_annot(annot_path)
        #> identify Brodmann parcels
        brodmann_parcel_indices = []
        for idx, name in enumerate(names):
            if 'Brodmann.' in name.decode():
                brodmann_parcel_indices.append(idx)
        #> Convert parcel id of non-Brodmann areas to 0
        brodmann_areas_mask = np.in1d(labels, np.array(brodmann_parcel_indices))
        labels_brodmann_only = np.where(brodmann_areas_mask, labels, 0)
        #> Change the parcel ids to range(0, n_Brodmann_parcels+1)
        parcels_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(np.unique(labels_brodmann_only))}
        labels = np.vectorize(parcels_new_idx.get)(labels_brodmann_only)
        #> Select ctab and names for the selected parcels (+ others: parcel id 0)
        included_parcel_indices = [0] + brodmann_parcel_indices
        ## parcel 93 (Brodmann.33) does not exist in the map => remove it
        included_parcel_indices = list(
            set(included_parcel_indices) \
            - (set(brodmann_parcel_indices) - set(np.unique(labels_brodmann_only)))
        )
        ctab = ctab[included_parcel_indices]
        names = np.array(names)[included_parcel_indices]
        nibabel.freesurfer.io.write_annot(
            annot_path.replace('_orig', ''),
            labels, ctab, names
        )