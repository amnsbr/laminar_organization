import os
import numpy as np
import pandas as pd
import scipy.stats
import nilearn.surface
import nibabel
import helpers
import enigmatoolbox.datasets

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')


def load_laminar_thickness(exc_masks=None, normalize_by_total_thickness=True, regress_out_curvature=False):
    """
    Loads laminar thickness data from 'data' folder and after masking out
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
            laminar_thickness[hem] = helpers.regress_out_surf_covariates(
                laminar_thickness[hem], cov_surf_data,
                sig_only=False, renormalize=True
                )
    return laminar_thickness

def calculate_alphas(betas, aw, ap):
    """
    Compute volume fraction map, alphas, that will yield the desired
    euclidean distance fraction map, betas, given vertex areas in the white matter surface, aw,
    and on the pial surface, ap.
    Based on Eq. 10 from https://doi.org/10.1016/j.neuroimage.2013.03.078 solved for alpha

    Parameters
    ----------
    betas: (np.ndarray) n_vertices x n_features (6 for layers), euclidean distance fraction map from wm surface
    aw: (np.ndarray) n_vertices, white matter surface area map
    ap: (np.ndarray) n_vertices, pial surface area map

    Returns
    ---------
    alphas: (np.ndarray) n_vertices x n_features (6 for layers), volume fraction map from wm surface
    """
    return ((aw + ((ap-aw)*betas))**2 - aw**2) / (ap**2 - aw**2)

def load_laminar_volume(exc_masks=None):
    """
    To correct for the effect of curvature on laminar thickness, this function 
    calculates relative volume of each layer at each vertex considering the
    relative thickness of that layer, and the curvature at that vertex.
    The conversion from relative thickness (or distance fraction, beta) 
    to volume fraction, alpha, is done by using Eq. 10 from 
    https://doi.org/10.1016/j.neuroimage.2013.03.078.
    This equation originally calculates beta given alpha, and was implemented in
    https://github.com/kwagstyl/surface_tools. To calculate alpha given beta, I've
    simply solved and rearranged Eq. 10 for alpha.

    This involves three steps:
    1. Convert relative laminar thickness arrays to betas, i.e., calculate cumulative relative thickness of layers from wm boundarty (i.e. beta6 would be thick6, beta5 would be thick6+thick5 and so on).
    2. Calculate the vertex-wise map of alphas for each layer (using vertex-wise pial and wm surface area calculated with CIVET in the original code & the function alpha). This will for each layer n be the volume of cortex from wm surface towards the layer boundary.
    3. Calculate the relative volume of each layer based on alphas.

    Parameters
    ----------
    exc_masks: (dict of str) Path to the surface masks of vertices that should be excluded (L and R) (format: .npy)

    Returns
    --------
    laminar_volume: (dict of np.ndarray) n_vertices x 6 for laminar volume of L and R hemispheres
    """
    # Load laminar thickness
    laminar_thickness = load_laminar_thickness(
        exc_masks=exc_masks, 
        normalize_by_total_thickness=True,
        regress_out_curvature=False
        )
    laminar_volume = {}
    for hem in ['L','R']:
        # Load wm and pial vertex areas
        wm_vertexareas = np.load(
            os.path.join(
                DATA_DIR, 'surface',
                f'tpl-bigbrain_hemi-{hem}_desc-white.area.npy'
            )
        )
        pia_vertexareas = np.load(
            os.path.join(
                DATA_DIR, 'surface',
                f'tpl-bigbrain_hemi-{hem}_desc-pial.area.npy'
            )
        )
        # Step 1: Convert relative laminar thickness and convert it to betas
        # (rho in Eq 10 or betas in the surface_tools)
        laminar_boundary_distance_from_wm = laminar_thickness[hem][:, ::-1].cumsum(axis=1)
        # Step 2: Convert relative distance from bottom (beta) to relative volume from bottom (alpha) based on curvature
        alphas = np.zeros_like(laminar_boundary_distance_from_wm)
        for lay_idx in range(6):
            alphas[:, lay_idx] = calculate_alphas(
                laminar_boundary_distance_from_wm[:, lay_idx], 
                wm_vertexareas, 
                pia_vertexareas
            )
        # Step 3: Convert cumulative fractional volume of stacked layers (alphas) to volume of individual layers
        laminar_volume[hem] = np.concatenate([
            alphas[:, 0][:, np.newaxis], # layer 6 is the first layer from bottom so its rel_vol = alpha
            np.diff(alphas, axis=1) # un-cumsum layers 5 to 1
            ], axis=1)[:,::-1] # reverse it from layer 6to1 to layer 1to6
    return laminar_volume

def load_laminar_density(exc_masks=None, method='mean'):
    """
    Loads laminar density data from 'src' folder, takes the average of sample densities
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
        #> labels post-processing for each specific parcellation
        if parcellation_name == 'sjh':
            sorted_labels = list(map(lambda l: int(l.decode().replace('sjh_','')), sorted_labels))
        elif parcellation_name == 'aparc':
            sorted_labels = list(map(lambda l: f'{hem}_{l.decode()}', sorted_labels)) # so that it matches ENIGMA toolbox dataset
        elif parcellation_name in ['schaefer400', 'economo']:
            sorted_labels = list(map(lambda l: l.decode(), sorted_labels)) # b'name' => 'name'
        transdict = dict(enumerate(sorted_labels))
        labeled_parcellation_map[hem] = np.vectorize(transdict.get)(parcellation_map)
    if concatenate:
        return np.concatenate([labeled_parcellation_map['L'], labeled_parcellation_map['R']])
    else:
        return labeled_parcellation_map

def load_cortical_types(parcellation_name=None):
    """
    Loads the surface map of cortical types parcellated or unparcellated

    Parameters
    ---------
    parcellation_name: (str or None)
        - None: vertex-wise
        - str: parcellation name (must exists in data/parcellation)
    """
    #> load the economo map and concatenate left and right hemispheres
    economo_maps = {}
    for hem in ['L', 'R']:
        economo_maps[hem] = nilearn.surface.load_surf_data(
            os.path.join(
                DATA_DIR, 'parcellation',
                f'tpl-bigbrain_hemi-{hem}_desc-economo_parcellation.label.gii'
                )
            )
    economo_map = np.concatenate([economo_maps['L'], economo_maps['R']])
    #> load the cortical types for each economo parcel
    economo_cortical_types = pd.read_csv(
        os.path.join(
            DATA_DIR, 'parcellated_surface',
            'economo_cortical_types.csv'
            )
        )
    economo_cortical_types.columns=['Label', 'Cortical Type']
    #> create the cortical types surface map
    cortical_types_map = economo_cortical_types.loc[economo_map, 'Cortical Type'].astype('category').reset_index(drop=True)
    cortical_types_map = cortical_types_map.cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
    if parcellation_name:
        #> load parcellation map
        parcellation_map = load_parcellation_map(parcellation_name, concatenate=True)
        parcellated_cortical_types_map = (
            #>> create a dataframe of surface map including both cortical type and parcel index
            pd.DataFrame({'Cortical Type': cortical_types_map, 'Parcel': pd.Series(parcellation_map)})
            #>> group by parcel
            .groupby('Parcel')
            #>> find the cortical types with the highest count
            ['Cortical Type'].value_counts(sort=True).unstack().idxmax(axis=1)
            #>> convert it back to category
            .astype('category')
            .cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
        )
        return parcellated_cortical_types_map
    else:
        # #> save the map
        # # TODO: maybe split hemispheres
        # np.save(
        #     os.path.join(
        #         DATA_DIR, 'parcellation',
        #         f'tpl-bigbrain_desc-cortical_types_parcellation.npy'
        #     ), cortical_types_map.cat.codes.values)
        return cortical_types_map      

def load_hist_mpc_gradients():
    """
    Loads two main gradients of Paquola et al. based on BigBrain MPC

    Returns
    -------
    hist_gradients: n_vertices x 2 (both hemispheres)
    """
    hist_gradients = {}
    for hist_gradient_num in range(1, 3):
        hist_gradients[hist_gradient_num] = {}
        for hem in ['L', 'R']:
            hist_gradients[hist_gradient_num][hem] = np.loadtxt(
                os.path.join(
                    DATA_DIR, 'surface',
                    f'tpl-bigbrain_hemi-{hem}_desc-Hist_G{hist_gradient_num}.txt'
                )
            )
        hist_gradients[hist_gradient_num] = np.concatenate([
            hist_gradients[hist_gradient_num]['L'], 
            hist_gradients[hist_gradient_num]['R']
            ])
    hist_gradients = np.vstack([
        hist_gradients[1],
        hist_gradients[2],
    ]).T
    return hist_gradients


def load_laminar_properties(regress_out_cruvature):
    """
    Loads laminar properties including relative thickness of each layer in addition to
    the sum of supragranular and infragranular layers and their ratios
    
    Parameters
    ----------
    regress_out_curvature: (bool) regress out curvature from relative laminar thickness

    Returns
    ------
    laminar_properties (pd.DataFrame) n_vert * n_properties
    """
    out_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_desc-laminarprops.csv')
    if os.path.exists(out_path):
        return pd.read_csv(out_path, sep=",", index_col=False)
    laminar_properties = helpers.read_laminar_thickness(regress_out_curvature=regress_out_cruvature)
    laminar_properties = pd.DataFrame(np.concatenate([laminar_properties['L'], laminar_properties['R']], axis=0))
    laminar_properties.columns = [f'Layer {idx+1}' for idx in range(6)]
    laminar_properties['L1-3 sum'] = laminar_properties.iloc[:, :3].sum(axis=1)
    laminar_properties['L4-6 sum'] = laminar_properties.iloc[:, 3:6].sum(axis=1)
    laminar_properties['L4-6 to L1-3 thickness ratio'] = laminar_properties.iloc[:, 3:6].sum(axis=1) / laminar_properties.iloc[:, :3].sum(axis=1)
    laminar_properties['L5-6 to L1-3 thickness ratio'] = laminar_properties.iloc[:, 4:6].sum(axis=1) / laminar_properties.iloc[:, :3].sum(axis=1)
    laminar_properties.to_csv(out_path, sep=',', index=False)
    return laminar_properties

def load_profile_moments():
    """
    Loads mean, std, skewness and kurtosis of vertical intensity profiles for all vertices

    Returns
    -------
    profile_moments (pd.DataFrame) n_vert * 4
    """
    out_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_desc-profilemoments.csv')
    if os.path.exists(out_path):
        return pd.read_csv(out_path, sep=",", index_col=False)
    profiles = np.loadtxt(os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_desc-profiles.txt'))
    profile_moments = pd.DataFrame()
    profile_moments['Intensity mean'] = np.mean(profiles, axis=0)
    profile_moments['Intensity std'] = np.std(profiles, axis=0)
    profile_moments['Intensity skewness'] = scipy.stats.skew(profiles, axis=0)
    profile_moments['Intensity kurtosis'] = scipy.stats.kurtosis(profiles, axis=0)
    profile_moments.to_csv(out_path, sep=',', index=False)
    return profile_moments

def load_disorder_atrophy_maps():
    out_path = os.path.join(DATA_DIR, 'surface', 'disorder_atrophy_maps.csv')
    if os.path.exists(out_path):
        return pd.read_csv(out_path, sep=",", index_col='Structure')
    disorder_atrophy_maps = pd.DataFrame()
    for disorder in ['adhd', 'bipolar', 'depression', 'ocd']:
        disorder_atrophy_maps[disorder] = (
            enigmatoolbox.datasets.load_summary_stats(disorder)
            ['CortThick_case_vs_controls_adult']
            .set_index('Structure')['d_icv'])
    disorder_atrophy_maps['schizophrenia'] = (
        enigmatoolbox.datasets.load_summary_stats('schizophrenia')
        ['CortThick_case_vs_controls']
        .set_index('Structure')['d_icv'])
    disorder_atrophy_maps['epilepsy'] = (
        enigmatoolbox.datasets.load_summary_stats('epilepsy')
        ['CortThick_case_vs_controls_allepilepsy']
        .set_index('Structure')['d_icv'])
    disorder_atrophy_maps.to_csv(out_path, sep=",", index_label='Structure')
    return disorder_atrophy_maps

def load_parcels_adys(parcellation_name):
    """
    Determines which parcels are in the adysgranular mask

    Parameter
    --------
    parcellation_name: (str) parcellation name (must exists in data/parcellation)

    Returns
    -------
    parcellated_mask: (dict of pd.Series) with n_parcels elements showing which parcels
        are in the adysgranular mask
    """
    adysgranular_masks = {
        'L': np.load(os.path.join(
            DATA_DIR, 'surface',
            f'tpl-bigbrain_hemi-L_desc-adysgranular_mask_parcellation-{parcellation_name}_thresh_0.1.npy'
        )),
        'R': np.load(os.path.join(
            DATA_DIR, 'surface',
            f'tpl-bigbrain_hemi-R_desc-adysgranular_mask_parcellation-{parcellation_name}_thresh_0.1.npy'
        ))
    }
    parcellated_mask = helpers.parcellate(adysgranular_masks, parcellation_name)
    return parcellated_mask
