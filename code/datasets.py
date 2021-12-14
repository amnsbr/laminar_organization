import os
import numpy as np
import pandas as pd
import scipy.stats
import nilearn.surface
import nibabel
import helpers
import enigmatoolbox.datasets
from helpers import DATA_DIR

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
        elif parcellation_name == 'schaefer400':
            sorted_labels = list(map(lambda l: l.decode(), sorted_labels)) # b'name' => 'name'
        elif parcellation_name == 'aparc':
            sorted_labels = list(map(lambda l: f'{hem}_{l.decode()}', sorted_labels)) # so that it matches ENIGMA toolbox dataset
        transdict = dict(enumerate(sorted_labels))
        labeled_parcellation_map[hem] = np.vectorize(transdict.get)(parcellation_map)
    if concatenate:
        return np.concatenate([labeled_parcellation_map['L'], labeled_parcellation_map['R']])
    else:
        return labeled_parcellation_map

def load_cortical_types_map():
    """
    Creates the surface map of cortical types (and saves it)
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
    #> save the map
    # TODO: maybe split hemispheres
    np.save(
        os.path.join(
            DATA_DIR, 'parcellation',
            f'tpl-bigbrain_desc-cortical_types_parcellation.npy'
        ), cortical_types_map.cat.codes.values)
    return cortical_types_map

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
