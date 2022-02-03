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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import brainspace.mesh, brainspace.plotting
import nilearn.surface
import nilearn.plotting
import statsmodels.api as sm
import datasets

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')
SRC_DIR = os.path.join(cwd, '..', 'src')
SPIN_BATCHES_DIR = os.path.join(SRC_DIR, 'spin_batches')
os.makedirs(SPIN_BATCHES_DIR, exist_ok=True)


MIDLINE_PARCELS = {
    'schaefer400': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'sjh': [0],
    'aparc': ['L_unknown', 'None']
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

def parcellate(surface_data, parcellation_name, averaging_method='median', na_midline=True):
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
        #> load parcellation map
        labeled_parcellation_maps = datasets.load_parcellation_map(parcellation_name, concatenate=False)
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
        #> load parcellation map
        labeled_parcellation_maps = datasets.load_parcellation_map(parcellation_name, concatenate=True)
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

def deparcellate(parcellated_data, parcellation_name):
    """
    Project the parcellated data to surface vertices.
    Note: only use this with parcellated data that has lost
    its parcel labels (e.g. surrogate maps)
    TODO: Ideally there should be no need for this to exist

    Parameters
    ----------
    parcellated_data: (pd.DataFrame | pd.Series) n_parcels x n_features | 1
    parcellation_name: (str)

    Returns
    -------
    surface_map: (np.ndarray) n_vertices [both hemispheres] x n_gradients
    """
    #> load concatenated parcellation map
    concat_parcellation_map = datasets.load_parcellation_map(parcellation_name, concatenate=True)
    #> load parcellated laminar data (we only need the index)
    dummy_surf_data = np.loadtxt(os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )
        )
    dummy_surf_data = np.zeros_like(dummy_surf_data)
    parcellated_dummy = parcellate(
        {'L': dummy_surf_data,
         'R': dummy_surf_data,},
         parcellation_name)
    concat_parcellated_dummy = concat_hemispheres(parcellated_dummy, dropna=False)
    all_parcels = concat_parcellated_dummy.index.to_series().rename('parcel')
    #> create a gradients dataframe including all parcels, where invalid parcels are NaN
    #   (this is necessary to be able to project it to the parcellation)
    labeled_parcellated_data = pd.concat(
        [
            parcellated_data,
            all_parcels
        ], axis=1).set_index('parcel')

    #>> label parcellated_data with index from the dummy parcellated data
    labeled_parcellated_data = pd.DataFrame(parcellated_data, index=concat_parcellated_dummy.index)
    #> get the map of gradients by indexing at parcellation labels
    surface_map = labeled_parcellated_data.loc[concat_parcellation_map].values # shape: vertex X gradient
    return surface_map


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

def plot_on_bigbrain_brainspace(surface_data_files, outfile=None):
    """
    Plots the `surface_data_files` on the bigbrain space and saves it in `outfile`
    using brainsapce.

    Note: Does not work in a remote server without proper X forwarding

    Parameters
    ----------
    surface_data_files: (dict of str) including paths to the surface file for 'L' and 'R' hemispheres
    outfile: (str) path to output; default would be the same as surface file
    """
    #> load bigbrain surfaces
    lh_surf = brainspace.mesh.mesh_io.read_surface(
        os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-mid.surf.gii')
        )
    rh_surf = brainspace.mesh.mesh_io.read_surface(
        os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-mid.surf.gii')
        )
    #> read surface data files and concatenate L and R
    if surface_data_files[0].endswith('.npy'):
        surface_data = np.concatenate([np.load(surface_data_files[0]), np.load(surface_data_files[1])])
    elif surface_data_files[0].endswith('.txt'):
        surface_data = np.concatenate([np.loadtxt(surface_data_files[0]), np.loadtxt(surface_data_files[1])])
    else:
        print("Surface data file not supported")
        return
    if not outfile:
        outfile = surface_data_files[0]+'.png'
    brainspace.plotting.surface_plotting.plot_hemispheres(lh_surf, rh_surf, surface_data,
        color_bar=True, interactive=False, embed_nb=False, size=(1600, 400), zoom=1.2,
        screenshot=True, filename=outfile, transparent_bg=True, offscreen=True)

def plot_on_bigbrain_nl(surface_data, filename, inflate=False, layout='horizontal', cmap='viridis'):
    """
    Plots the `surface_data_files` on the bigbrain space and saves it in `outfile`
    using nilearn

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) (n_vert,) surface data: concatenated or 'L' and 'R' hemispheres
    filename: (str) path to output; default would be the same as surface file
    inflate: (bool) whether to plot the inflated surface
    layout:
        - horizontal: left-lateral, left-medial, right-medial, right-lateral
        - grid: lateral views on the top and medial views on the bottom
    cmap: (str)
    """
    #> split surface if it has been concatenated (e.g. gradients)
    #  and make sure the shape is correct
    n_hem_vertices = np.loadtxt(
        os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )
        ).size
    if isinstance(surface_data, np.ndarray):
        assert surface_data.shape[0] == n_hem_vertices * 2
        lh_surface_data = surface_data[:n_hem_vertices]
        rh_surface_data = surface_data[n_hem_vertices:]
        surface_data = {'L': lh_surface_data, 'R': rh_surface_data}
    else:
        assert surface_data['L'].shape[0] == n_hem_vertices
    #> initialize the figures
    if layout == 'horizontal':
        figure, axes = plt.subplots(1, 4, figsize=(24, 5), subplot_kw={'projection': '3d'})
    elif layout == 'grid':
        figure, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten(order='F')
    curr_ax_idx = 0
    for hemi in ['left', 'right']:
        #> plot the medial and lateral views
        if hemi == 'left':
            views_order = ['lateral', 'medial']
            mesh_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-mid.surf.gii')
        else:
            views_order = ['medial', 'lateral']
            mesh_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-mid.surf.gii')
        if inflate:
            mesh_path = mesh_path.replace('.surf', '.surf.inflate')
        for view in views_order:
            nilearn.plotting.plot_surf(
                mesh_path,
                surface_data[hemi[0].upper()],
                hemi=hemi, view=view, axes=axes[curr_ax_idx],
                cmap=cmap
            )
            curr_ax_idx += 1
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename, dpi=192)

#### Spin permutation functions ####
def create_bigbrain_spin_permutations(n_perm=1000, batch_size=20):
    """
    Creates spin permutations of the bigbrain surface sphere and stores the
    vertex indices of spins in batches on 'src' folder

    n_perm: (int) total number of permutations
    batch_size: (int) number of permutations per batch
    """
    print(f"Creating {n_perm} spin permutations")
    if os.path.exists(os.path.join(SPIN_BATCHES_DIR, 'tpl-bigbrain_desc-spin_indices_batch0.npz')):
        print("Spin permutation batches already exist")
        return
    #> read the bigbrain surface sphere files as a mesh that can be used by spin_permutations function
    lh_sphere = brainspace.mesh.mesh_io.read_surface(os.path.join(DATA_DIR, 'surface', f'tpl-bigbrain_hemi-L_desc-sphere_rot_fsaverage.surf.gii'))
    rh_sphere = brainspace.mesh.mesh_io.read_surface(os.path.join(DATA_DIR, 'surface', f'tpl-bigbrain_hemi-R_desc-sphere_rot_fsaverage.surf.gii'))
    #> create permutations of surface_data with preserved spatial-autocorrelation using spin_permutations
    #  doing it in batches to reduce memory requirements
    n_batch = n_perm // batch_size
    for batch in range(n_batch):
        print('\t\tBatch', batch)
        spin_idx = brainspace.null_models.spin._generate_spins(
            lh_sphere, rh_sphere, 
            n_rep = batch_size,
            random_state = 9*batch, # it's important for random states to be different across batches
            )
        np.savez_compressed(os.path.join(SPIN_BATCHES_DIR, f'tpl-bigbrain_desc-spin_indices_batch{batch}.npz'), lh=spin_idx['lh'], rh=spin_idx['rh'])

def spin_test(surface_data_to_spin, surface_data_target):
    """
    Performs spin test on the correlation between `surface_data_to_spin` and 
    `surface_data_target` after parcellation, where `surface_data_to_spin` is spun

    Parameters
    ----------
    surface_data_to_spin: (np.ndarray | dict of np.ndarray) n_vert * n_features in bigbrain surface space of L and R hemispheres
    surface_data_target: (np.ndarray | dict of np.ndarray) n_vert * n_features in bigbrain surface space of L and R hemispheres
    n_perm: (int) number of spin permutations
    batch_prefix: (str) batches filename prefix
    """
    #> create spin permutation batches
    create_bigbrain_spin_permutations()
    #> split hemispheres
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
    for batch_file in sorted(glob.glob(os.path.join(SPIN_BATCHES_DIR, f'tpl-bigbrain_desc-spin_indices_batch*.npz'))):
        print("\t\tBatch", batch_file)
        #> load the 20-spin batch of spin permutated maps and concatenate left and right hemispheres
        batch_idx = np.load(batch_file) # n_perm * n_vert arrays for 'lh' and 'rh'
        batch_lh_surrogates = surface_data_to_spin['L'][batch_idx['lh']] # n_perm * n_vert * n_features
        batch_rh_surrogates = surface_data_to_spin['R'][batch_idx['rh']]
        concat_batch_surrogates = np.concatenate([batch_lh_surrogates, batch_rh_surrogates], axis=1)
        for perm_idx in range(batch_rh_surrogates.shape[0]):
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
            #> free up memory
            del surrogate
            gc.collect()
    #> remove the test_r from null_distribution
    null_distribution = null_distribution[1:, :, :]
    #> calculate p value
    p_val = (np.abs(null_distribution) >= np.abs(test_r)).mean(axis=0)
    #> reduce unnecessary dimension of test_r
    test_r = test_r[0, :, :]
    return test_r, p_val, null_distribution