import os
import numpy as np
import pandas as pd
import scipy.stats
import nilearn.surface
import nibabel
import helpers
import enigmatoolbox.datasets
from cortex.polyutils import Surface
import scipy.spatial.distance
import scipy.io

import helpers


#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')

#> constants / configs
N_VERTICES_HEM_BB = 163842
N_VERTICES_HEM_BB_ICO5 = 10242

def load_downsampled_surface_paths(kind='orig'):
    """
    Loads or creates downsampled surfaces of bigbrain left and right hemispheres
    (from ico7 to ico5)
    The actual downsampling is done in `local/downsample_bb.m` in matlab and this
    function just makes it python-readable

    Parameters
    ----------
    kind: (str)
        - 'orig'
        - 'inflated'
        - 'sphere'

    Returns
    ------
    paths: (dict of str) path to downsampled surfaces of L and R
    """
    paths = {}
    for hem in ['L', 'R']:
        paths[hem] = os.path.join(
            SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled_{kind}.surf.gii'
            )
        if os.path.exists(paths[hem]):
            continue
        #> load the faces and coords of downsampled surface from matlab results
        mat = scipy.io.loadmat(
            os.path.join(
                SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
        )
        faces = mat['BB10'][0,0][0]-1 # 1-indexing of matlab to 0-indexing of python
        coords = mat['BB10'][0,0][1].T
        if kind != 'orig':
            #> load the sphere|inflated ico7 surface to use their coordinates
            if kind == 'sphere':
                deformed_path = os.path.join(SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-sphere_rot_fsaverage.surf.gii')
            else:
                deformed_path = os.path.join(SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.inflate.gii')
            deformed_ico7 = nilearn.surface.load_surf_mesh(deformed_path)
            #> get the coordinates of central vertices which are in both ico7 and ico5.
            #  Note that for faces we will use the faces of non-deformed version of ico5
            #  surface, as it wouldn be the same regardless of the shape of surface (i.e.,
            #  the position of vertices)
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            coords = deformed_ico7.coordinates[bb_downsample_indices]
        #> save it as gifti
        bb10_gifti = nibabel.gifti.gifti.GiftiImage(
            darrays = [
                nibabel.gifti.gifti.GiftiDataArray(
                    data=coords,
                    intent='NIFTI_INTENT_POINTSET'
                    ),
                nibabel.gifti.gifti.GiftiDataArray(
                    data=faces,
                    intent='NIFTI_INTENT_TRIANGLE'
                    )
            ])
        nibabel.save(bb10_gifti, paths[hem])
    return paths



def load_curvature_maps(downsampled=False):
    """
    Creates the map of curvature for the bigbrain surface
    using pycortex or loads it if it already exist

    Returns
    -------
    curvature_maps: (dict of np.ndarray) n_vertices, for L and R hemispheres
    """
    curvature_maps = {}
    for hem in ['L', 'R']:
        curvature_filepath = os.path.join(
            OUTPUT_DIR, 'curvature', 
            f'tpl-bigbrain_hemi-{hem}'\
            + f'_desc-{"downsampled_" if downsampled else ""}mean_curvature.npy'
            )
        if os.path.exists(curvature_filepath):
            curvature_maps[hem] = np.load(curvature_filepath)
            continue
        #> load surface
        if downsampled:
            mesh_path = load_downsampled_surface_paths('orig')[hem]
        else:
            mesh_path = os.path.join(
                SRC_DIR,
                f'tpl-bigbrain_hemi-{hem}_desc-pial.surf.gii'
                )
        vertices, faces = nilearn.surface.load_surf_mesh(mesh_path)
        surface = Surface(vertices, faces)
        #> calculate mean curvature
        curvature = surface.mean_curvature()
        #> save it
        os.makedirs(os.path.join(OUTPUT_DIR, 'curvature'), exist_ok=True)
        np.save(curvature_filepath, curvature)
        curvature_maps[hem] = curvature
    return curvature_maps

def load_cortical_types(parcellation_name=None, downsampled=False):
    """
    Loads the surface map of cortical types parcellated or unparcellated

    Parameters
    ---------
    parcellation_name: (str or None)
        - None: vertex-wise
        - str: parcellation name (must exists in data/parcellation)
    downsampled: (bool) load downsampled map of cortical types

    Returns
    --------
    cortical_types_map (pd.DataFrame): with the length n_vertices|n_parcels
    """
    #> load the economo map and concatenate left and right hemispheres
    economo_maps = {}
    for hem in ['L', 'R']:
        economo_maps[hem] = nilearn.surface.load_surf_data(
            os.path.join(
                SRC_DIR,
                f'tpl-bigbrain_hemi-{hem}_desc-economo_parcellation.label.gii'
                )
            )
        if downsampled:
            #> select only the vertices corresponding to the downsampled ico5 surface
            mat = scipy.io.loadmat(
                os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
            )
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            economo_maps[hem] = economo_maps[hem][bb_downsample_indices]
    economo_map = np.concatenate([economo_maps['L'], economo_maps['R']])
    #> load the cortical types for each economo parcel
    economo_cortical_types = pd.read_csv(
        os.path.join(
            SRC_DIR,
            'economo_cortical_types.csv'
            )
        )
    economo_cortical_types.columns=['Label', 'Cortical Type']
    #> create the cortical types surface map
    cortical_types_map = economo_cortical_types.loc[economo_map, 'Cortical Type'].astype('category').reset_index(drop=True)
    cortical_types_map = cortical_types_map.cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
    if parcellation_name:
        #> load parcellation map
        parcellation_map = load_parcellation_map(parcellation_name, concatenate=True, downsampled=downsampled)
        parcellated_cortical_types_map = (
            #>> create a dataframe of surface map including both cortical type and parcel index
            pd.DataFrame({'Cortical Type': cortical_types_map, 'Parcel': pd.Series(parcellation_map)})
            #>> group by parcel
            .groupby('Parcel')
            #>> find the cortical types with the highest count (may also be non-cortex)
            ['Cortical Type'].value_counts(sort=True).unstack().idxmax(axis=1)
            #>> convert it back to category
            .astype('category')
            .cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
        )
        #> assign cortical types of midline parcels to NaN
        parcellated_cortical_types_map.loc[helpers.MIDLINE_PARCELS[parcellation_name]] = np.NaN
        return parcellated_cortical_types_map
    else:
        return cortical_types_map      

def load_exc_masks(exc_mask_type, parcellation_name=None):
    """
    Create masks of bigbrain space including agranular and dysgranular region.
    When the data is processed parcellated, these masks also take the parcellation
    into account and include parcels in which the most common cortical type is 
    non-cortical, allocortex, +/- a-/dysgranular regions

    Parameters
    ---------
    exc_mask_type: (str)
        - allocortex: excludes allocortex
        - adysgranular: excludes allocortex + adysgranular regions
    parcellation_name: (None or str) the parcellation should be saved in src folder
                        with the name 'tpl-bigbrain_hemi-L_desc-{parcellation_name}_parcellation.label.gii'

    Returns
    -------
    hem_exc_masks: (dict) boolean surface maps of exclusion masks for L and R hemispheres
    """
    mask_filepath = os.path.join(
        OUTPUT_DIR, 'masks',
        f'tpl-bigbrain_desc-{exc_mask_type}_mask_parc-{parcellation_name}.npy'
    )
    if os.path.exists(mask_filepath):
        exc_mask = np.load(mask_filepath)
    else:
        #> load cortical types surface map after/without parcellation
        cortical_types = load_cortical_types(parcellation_name)
        #> if it's parcellated project back to surface
        #  Why? because parcellation_map and economo parcellation do
        #  not fully overlap and we need a mask that is aligned with the parcellation_map
        if parcellation_name: 
            cortical_types = pd.Series(helpers.deparcellate(cortical_types, parcellation_name).flatten())
        #> match midlines of the `parcellation_name` and economo parcellation (sometimes e.g. in sjh
        #  they do not match) by makign their overlap NaN. The midline of `parcellation_name` is already
        #  NaN (from load_cortical_types). Therefore only load midline of economo and make it NaN
        economo_map = load_parcellation_map('economo', concatenate=True)
        economo_midline_mask = np.in1d(economo_map, ['unknown', 'corpuscallosum'])
        cortical_types[economo_midline_mask] = np.NaN
        #> create a mask of excluded cortical types
        if exc_mask_type == 'allocortex':
            exc_cortical_types = [np.NaN, 'ALO']
        elif exc_mask_type == 'adysgranular':
            exc_cortical_types = [np.NaN, 'ALO', 'DG', 'AG']
        exc_mask = cortical_types.isin(exc_cortical_types).values
        os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)
        np.save(mask_filepath, exc_mask)
    #> for historical reasons split this into two hemispheres and return it
    hem_exc_masks = {
        'L': exc_mask[:N_VERTICES_HEM_BB],
        'R': exc_mask[N_VERTICES_HEM_BB:]
    }
    return hem_exc_masks

def load_laminar_thickness(exc_masks=None, normalize_by_total_thickness=True, regress_out_curvature=False):
    """
    Loads laminar thickness data from 'data' folder and after masking out
    `exc_mask` returns 6-d laminar thickness arrays for left and right hemispheres.
    Also does normalization if `self.normalize_by_total_thickness` is True.

    Parameters
    --------
    exc_masks: (dict of np.ndarray) The surface masks of vertices that should be excluded (L and R)
    normalize_by_total_thickness: (bool) Normalize by total thickness. Default: True
    regress_out_curvature: (bool) Regress out curvature. Default: False

    Retruns
    --------
    laminar_thickness: (dict of np.ndarray) n_vertices x 6 for laminar thickness of L and R hemispheres
    """
    laminar_thickness = {}
    for hem in ['L', 'R']:
        #> read the laminar thickness data from bigbrainwrap .txt files
        laminar_thickness[hem] = np.empty((N_VERTICES_HEM_BB, 6))
        for layer_num in range(1, 7):
            laminar_thickness[hem][:, layer_num-1] = np.loadtxt(
                os.path.join(
                    SRC_DIR,
                    f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
                    ))
        #> remove the exc_mask
        if exc_masks:
            laminar_thickness[hem][exc_masks[hem], :] = np.NaN
        #> normalize by total thickness
        if normalize_by_total_thickness:
            laminar_thickness[hem] /= laminar_thickness[hem].sum(axis=1, keepdims=True)
        #> regress out curvature
        if regress_out_curvature:
            cov_surf_data = load_curvature_maps()[hem]
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
    exc_masks: (dict of np.ndarray) The surface masks of vertices that should be excluded (L and R)

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
                SRC_DIR,
                f'tpl-bigbrain_hemi-{hem}_desc-white.area.npy'
            )
        )
        pia_vertexareas = np.load(
            os.path.join(
                SRC_DIR,
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

def load_total_depth_density(exc_masks=None):
    """
    Loads laminar density of total cortical depth sampled at 50 points and after masking out
    `exc_masks` returns separate arrays for L and R hemispheres

    Parameters
    ---------
    exc_masks: (dict of np.ndarray) The surface masks of vertices that should be excluded (L and R)
 

    Returns
    -------
    density_profiles (dict of np.ndarray) n_vert x 50 for L and R hemispheres
    """
    #> load profiles and reshape to n_vert x 50
    density_profiles = np.loadtxt(os.path.join(SRC_DIR, 'tpl-bigbrain_desc-profiles.txt'))
    density_profiles = density_profiles.T
    #> split hemispheres
    density_profiles = {
        'L': density_profiles[:density_profiles.shape[0]//2, :],
        'R': density_profiles[density_profiles.shape[0]//2:, :],
    }
    #> remove the exc_mask
    if exc_masks:
        for hem in ['L', 'R']:
            density_profiles[hem][exc_masks[hem], :] = np.NaN

    return density_profiles


def load_laminar_density(exc_masks=None, method='mean'):
    """
    Loads laminar density data from 'src' folder, takes the average of sample densities
    for each layer, and after masking out `exc_mask` returns 6-d average laminar density 
    arrays for left and right hemispheres

    Parameters
    ----------
    exc_masks: (dict of np.ndarray) The surface masks of vertices that should be excluded (L and R)
    method: (str) method of finding central tendency of samples in each layer in each vertex
        - mean
        - median

    Retruns
    --------
    laminar_density: (dict of np.ndarray) n_vertices x 6 for laminar density of L and R hemispheres
    """
    laminar_density = {}
    for hem in ['L', 'R']:
        #> read the laminar thickness data from bigbrainwrap .txt files
        laminar_density[hem] = np.empty((N_VERTICES_HEM_BB, 6))
        for layer_num in range(1, 7):
            profiles = np.load(
                os.path.join(
                    SRC_DIR,
                    f'tpl-bigbrain_hemi-{hem[0].upper()}_desc-layer{layer_num}_profiles_nsurf-10.npz'
                    ))['profiles']
            if method == 'mean':
                laminar_density[hem][:, layer_num-1] = profiles.mean(axis=0)
            elif method == 'median':
                laminar_density[hem][:, layer_num-1] = np.median(profiles, axis=0)
        #> remove the exc_mask
        if exc_masks:
            laminar_density[hem][exc_masks[hem], :] = np.NaN
        # TODO: also normalize density?
    return laminar_density

def load_parcellation_map(parcellation_name, concatenate, downsampled=False):
    """
    Loads parcellation maps of L and R hemispheres, correctly relabels them
    and concatenates them if `concatenate` is True
    TODO: maybe add the option for loading only the parcel indices

    Parameters
    ----------
    parcellation_name: (str) Parcellation scheme
    concatenate: (bool) whether to cocnatenate the hemispheres
    downsampled: (bool) whether to load a downsampled version of parcellation map

    Returns
    -------
    labeled_parcellation_map: (np.ndarray or dict of np.ndarray) concatenated or unconcatenated labeled parcellation map
    """
    labeled_parcellation_map = {}
    for hem in ['L', 'R']:
        #> load parcellation map
        parcellation_map = nilearn.surface.load_surf_data(
            os.path.join(
                SRC_DIR, 
                f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation.label.gii')
            )
        #> label parcellation map
        _, _, sorted_labels = nibabel.freesurfer.io.read_annot(
            os.path.join(
                SRC_DIR, 
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
        if downsampled:
            #> select only the vertices corresponding to the downsampled ico5 surface
            # Warning: For fine grained parcels such as sjh this approach leads to
            # a few parcels being removed
            mat = scipy.io.loadmat(
                os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
            )
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            labeled_parcellation_map[hem] = labeled_parcellation_map[hem][bb_downsample_indices]
    if concatenate:
        return np.concatenate([labeled_parcellation_map['L'], labeled_parcellation_map['R']])
    else:
        return labeled_parcellation_map


def load_parcels_adys(parcellation_name, concat=True):
    """
    Determines which parcels are in the adysgranular mask

    Parameter
    --------
    parcellation_name: (str) parcellation name (must exists in data/parcellation)
    concat: (bool) concatenate hemispheres

    Returns
    -------
    parcels_adys: (pd.Series | dict of pd.Series) with n_parcels elements showing which parcels
        are in the adysgranular mask
    """
    adysgranular_masks = load_exc_masks('adysgranular', parcellation_name)
    parcels_adys = helpers.parcellate(adysgranular_masks, parcellation_name)
    if concat:
        parcels_adys = helpers.concat_hemispheres(parcels_adys, dropna=True)
    return parcels_adys

def load_disorder_maps():
    """
    Loads maps of cortical thickness difference in disorders.
    Adult popultation, mega-analyses, and more general categories
    of disorders (e.g. all epliepsy vs TLE) are preferred

    Returns
    -------
    parcellated_disorder_maps (pd.DataFrame): 
    """
    parcellated_disorder_maps = pd.DataFrame()
    for disorder in ['adhd', 'bipolar', 'depression', 'ocd']:
        parcellated_disorder_maps[disorder] = (
            enigmatoolbox.datasets.load_summary_stats(disorder)
            ['CortThick_case_vs_controls_adult']
            .set_index('Structure')['d_icv'])
    parcellated_disorder_maps['schizophrenia'] = (
        enigmatoolbox.datasets.load_summary_stats('schizophrenia')
        ['CortThick_case_vs_controls']
        .set_index('Structure')['d_icv'])
    parcellated_disorder_maps['asd'] = (
        enigmatoolbox.datasets.load_summary_stats('asd')
        ['CortThick_case_vs_controls_mega_analysis']
        .set_index('Structure')['d_icv'])
    parcellated_disorder_maps['epilepsy'] = (
        enigmatoolbox.datasets.load_summary_stats('epilepsy')
        ['CortThick_case_vs_controls_allepilepsy']
        .set_index('Structure')['d_icv'])
    return parcellated_disorder_maps


def load_conn_matrices(kind, parcellation_name='schaefer400'):
    """
    Loads FC or SC matrices in Schaefer parcellation (400) from ENIGMA toolbox
    and reorders it according to `matrix_file`. For SC matrix also makes contralateral
    values 0 (so that they are not included in correlations)

    Parameters
    ----------
    kind: (str)
        - structural
        - functional
    
    parcellation_name: (str)
        - schaefer400 (currently only this is supported)

    Returns
    ---------
    reordered_conn_matrix, (np.ndarray) (n_parc, n_parc) 
        reordered FC or SC matrices matching the original matrix
    """
    #> match parcellation name with the enigmatoolbox
    ENIGMATOOLBOX_PARC_NAMES = {
        'schaefer400': 'schaefer_400'
    }
    enigma_parcellation_name = ENIGMATOOLBOX_PARC_NAMES.get(parcellation_name, None)
    if enigma_parcellation_name == None:
        raise Exception(f"ENIGMA Toolbox does not have connectivity data for the parcellation {parcellation_name}")
    #> load data from enigma toolbox
    if kind == 'structural':
        conn_matrix, conn_matrix_labels, _, _ = enigmatoolbox.datasets.load_sc(enigma_parcellation_name)
    else:
        conn_matrix, conn_matrix_labels, _, _ = enigmatoolbox.datasets.load_fc(enigma_parcellation_name)
    conn_matrix = pd.DataFrame(conn_matrix, columns=conn_matrix_labels, index=conn_matrix_labels)
    #> parcellate dummy data to get the order of parcels in other matrices created locally
    dummy_surf_data = np.zeros(N_VERTICES_HEM_BB * 2)
    parcellated_dummy = helpers.parcellate(dummy_surf_data, parcellation_name).dropna()
    #> reorder matrices downloaded from enigmaltoolbox
    reordered_conn_matrix = conn_matrix.loc[parcellated_dummy.index, parcellated_dummy.index]
    return reordered_conn_matrix
