import urllib.request
import shutil
from urllib.parse import urlparse
import os
from ftplib import FTP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import brainspace.mesh, brainspace.plotting
import nilearn.surface
import nilearn.plotting
import nibabel.freesurfer.io
import statsmodels.api as sm

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')

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
def read_laminar_thickness(exc_masks=None, normalize_by_total_thickness=True, regress_out_curvature=False):
    """
    Reads laminar thickness data from 'data' folder and after masking out
    `exc_mask` returns 6-d laminar thickness arrays for left and right hemispheres.
    Also does normalization if `self.normalize_by_total_thickness` is True.

    Parameters
    --------
    exc_masks: (dict of str) Path to the surface masks of vertices that should be excluded (L and R) (format: .npy)
    normalize_by_total_thickness: (bool) Normalize by total thickness. Default: True
    regress_out_curvature: (bool) Regress out curvature. Default: False

    Retruns
    --------
    laminar_thickness: (dict of np.ndarray) n_vertices x 6 for laminar thickness of L and R hemispheres
    """
    #> get the number of vertices
    n_hem_vertices = np.loadtxt(
        os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )
        ).size
    laminar_thickness = {}
    for hem in ['L', 'R']:
        #> read the laminar thickness data from bigbrainwrap .txt files
        laminar_thickness[hem] = np.empty((n_hem_vertices, 6))
        for layer_num in range(1, 7):
            laminar_thickness[hem][:, layer_num-1] = np.loadtxt(
                os.path.join(
                    DATA_DIR, 'surface',
                    f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
                    ))
        #> remove the exc_mask
        if exc_masks:
            exc_mask_map = np.load(exc_masks[hem])
            laminar_thickness[hem][exc_mask_map, :] = np.NaN
        #> normalize by total thickness
        if normalize_by_total_thickness:
            laminar_thickness[hem] /= laminar_thickness[hem].sum(axis=1, keepdims=True)
        #> regress out curvature
        if regress_out_curvature:
            cov_surf_data = np.load(
                os.path.join(
                    DATA_DIR, 'surface', 
                    f'tpl-bigbrain_hemi-{hem}_desc-mean_curvature.npy'
                    ))
            laminar_thickness[hem] = regress_out_surf_covariates(
                laminar_thickness[hem], cov_surf_data,
                sig_only=False, renormalize=True
                )
    return laminar_thickness

def read_laminar_density(exc_masks=None, method='mean'):
    """
    Reads laminar density data from 'src' folder, takes the average of sample densities
    for each layer, and after masking out `exc_mask` returns 6-d average laminar density 
    arrays for left and right hemispheres

    Parameters
    ----------
    exc_masks: (dict of str) Path to the surface masks of vertices that should be excluded (L and R) (format: .npy)
    method: (str) method of finding central tendency of samples in each layer in each vertex
        - mean
        - median

    Retruns
    --------
    laminar_density: (dict of np.ndarray) n_vertices x 6 for laminar density of L and R hemispheres
    """
    #> get the number of vertices
    n_hem_vertices = np.loadtxt(
        os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )
        ).size
    laminar_density = {}
    for hem in ['L', 'R']:
        #> read the laminar thickness data from bigbrainwrap .txt files
        laminar_density[hem] = np.empty((n_hem_vertices, 6))
        for layer_num in range(1, 7):
            profiles = np.load(
                os.path.join(
                    DATA_DIR, 'surface',
                    f'tpl-bigbrain_hemi-{hem[0].upper()}_desc-layer-{layer_num}_profiles_nsurf-10.npz'
                    ))['profiles']
            if method == 'mean':
                laminar_density[hem][:, layer_num-1] = profiles.mean(axis=0)
            elif method == 'median':
                laminar_density[hem][:, layer_num-1] = np.median(profiles, axis=0)
        #> remove the exc_mask
        if exc_masks:
            exc_mask_map = np.load(exc_masks[hem])
            laminar_density[hem][exc_mask_map, :] = np.NaN
        # TODO: also normalize density?
    return laminar_density

def load_parcellation_map(parcellation_name, concatenate):
    """
    Loads parcellation maps of L and R hemispheres, correctly relabels them
    and concatenates them if `concatenate` is True

    Parameters
    ----------
    parcellation_name: (str) Parcellation scheme
    concatenate: (bool) whether to cocnatenate the hemispheres

    Returns
    -------
    labeled_parcellation_map: (np.ndarray or dict of np.ndarray) concatenated or unconcatenated labeled parcellation map
    """
    labeled_parcellation_map = {}
    for hem in ['L', 'R']:
        #> load parcellation map
        parcellation_map = nilearn.surface.load_surf_data(
            os.path.join(
                DATA_DIR, 'parcellation', 
                f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation.label.gii')
            )
        #> label parcellation map
        _, _, sorted_labels = nibabel.freesurfer.io.read_annot(
            os.path.join(
                DATA_DIR, 'parcellation', 
                f'{hem.lower()}h_{parcellation_name}.annot')
        )
        if parcellation_name == 'sjh':
            sorted_labels = list(map(lambda l: int(l.decode().replace('sjh_','')), sorted_labels))
        transdict = dict(enumerate(sorted_labels))
        labeled_parcellation_map[hem] = np.vectorize(transdict.get)(parcellation_map)
    if concatenate:
        return np.concatenate([labeled_parcellation_map['L'], labeled_parcellation_map['R']])
    else:
        return labeled_parcellation_map

def parcellate(surface_data, parcellation_name, averaging_method='median'):
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

    Returns
    ---------
    parcellated_data: (pd.DataFrame or dict of pd.DataFrame) n_parcels x n_features for data of L and R hemispheres or both hemispheres
    """
    if isinstance(surface_data, dict):
        #> load parcellation map
        labeled_parcellation_maps = load_parcellation_map(parcellation_name, concatenate=False)
        parcellated_data = {}
        for hem in ['L', 'R']:
            #> parcellate
            parcellated_vertices = (
                pd.DataFrame(surface_data[hem], index=labeled_parcellation_maps[hem])
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
        labeled_parcellation_maps = load_parcellation_map(parcellation_name, concatenate=True)
        #> parcellate
        parcellated_vertices = (
            pd.DataFrame(surface_data, index=labeled_parcellation_maps)
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
    parcellated_data: (np.ndarray) n_parcels x n_features
    parcellation_name: (str)

    Returns
    -------
    surface_map: (np.ndarray) n_vertices [both hemispheres] x n_gradients
    """
    #> load concatenated parcellation map
    concat_parcellation_map = load_parcellation_map(parcellation_name, concatenate=True)
    #> load parcellated laminar data (we only need the index)
    dummy_surf_data = np.loadtxt(os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )
        )
    parcellated_dummy = parcellate(
        {'L': dummy_surf_data,
         'R': dummy_surf_data,},
         parcellation_name)
    concat_parcellated_dummy = concat_hemispheres(parcellated_dummy, dropna=False)
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

def plot_on_bigbrain_nl(surface_data, filename):
    """
    Plots the `surface_data_files` on the bigbrain space and saves it in `outfile`
    using nilearn

    Parameters
    ----------
    surface_data: (np.ndarray or dict of np.ndarray) (n_vert,) surface data: concatenated or 'L' and 'R' hemispheres
    outfile: (str) path to output; default would be the same as surface file
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
    figure, axes = plt.subplots(1, 4, figsize=(24, 5), subplot_kw={'projection': '3d'})
    curr_ax_idx = 0
    for hemi in ['left', 'right']:
        #> plot the medial and lateral views
        if hemi == 'left':
            views_order = ['lateral', 'medial']
            mesh_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-mid.surf.gii')
        else:
            views_order = ['medial', 'lateral']
            mesh_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-mid.surf.gii')
        for view in views_order:
            nilearn.plotting.plot_surf(
                mesh_path,
                surface_data[hemi[0].upper()],
                hemi=hemi, view=view, axes=axes[curr_ax_idx],
            )
            curr_ax_idx += 1
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename, dpi=192)
