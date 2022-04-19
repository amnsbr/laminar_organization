import os
import logging
import sys
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
import abagen
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import math_img
import netneurotools.datasets


import helpers

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
ABAGEN_DIR = '/data/group/cng/abagen-data'

# constants / configs
N_VERTICES_HEM_BB = 163842
N_VERTICES_HEM_BB_ICO5 = 10242

LAYERS_COLORS = {
    'bigbrain': ['#abab6b', '#dabcbc', '#dfcbba', '#e1dec5', '#66a6a6','#d6c2e3'], # layer 1 to 6
    'wagstyl': ['#3a6aa6ff', '#f8f198ff', '#f9bf87ff', '#beaed3ff', '#7fc47cff','#e31879ff']
}

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
        # load the faces and coords of downsampled surface from matlab results
        mat = scipy.io.loadmat(
            os.path.join(
                SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
        )
        faces = mat['BB10'][0,0][0]-1 # 1-indexing of matlab to 0-indexing of python
        coords = mat['BB10'][0,0][1].T
        if kind != 'orig':
            # load the sphere|inflated ico7 surface to use their coordinates
            if kind == 'sphere':
                deformed_path = os.path.join(SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-sphere_rot_fsaverage.surf.gii')
            else:
                deformed_path = os.path.join(SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.inflate.gii')
            deformed_ico7 = nilearn.surface.load_surf_mesh(deformed_path)
            # get the coordinates of central vertices which are in both ico7 and ico5.
            #  Note that for faces we will use the faces of non-deformed version of ico5
            #  surface, as it wouldn be the same regardless of the shape of surface (i.e.,
            #  the position of vertices)
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            coords = deformed_ico7.coordinates[bb_downsample_indices]
        # save it as gifti
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
        # load surface
        if downsampled:
            mesh_path = load_downsampled_surface_paths('orig')[hem]
        else:
            mesh_path = os.path.join(
                SRC_DIR,
                f'tpl-bigbrain_hemi-{hem}_desc-pial.surf.gii'
                )
        vertices, faces = nilearn.surface.load_surf_mesh(mesh_path)
        surface = Surface(vertices, faces)
        # calculate mean curvature
        curvature = surface.mean_curvature()
        # save it
        os.makedirs(os.path.join(OUTPUT_DIR, 'curvature'), exist_ok=True)
        np.save(curvature_filepath, curvature)
        curvature_maps[hem] = curvature
    return curvature_maps

def load_cortical_types(parcellation_name=None, downsampled=False):
    """
    Loads the map of cortical types

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
    # load the economo map
    economo_map = load_parcellation_map('economo', True, downsampled=downsampled)
    # load the cortical types for each economo parcel
    economo_cortical_types = pd.read_csv(
        os.path.join(
            SRC_DIR,
            'economo_cortical_types.csv'
            ))
    # duplicate the economo parcel to cortical type mapping for each hemisphere
    hemis = pd.Series(['L_']*economo_cortical_types.shape[0]+['R_']*economo_cortical_types.shape[0])
    economo_cortical_types = pd.concat([economo_cortical_types]*2, axis=0).reset_index(drop=True)
    economo_cortical_types['Label'] = hemis + economo_cortical_types['Label']
    economo_cortical_types = economo_cortical_types.set_index('Label')
    # create the cortical types surface map
    cortical_types_map = economo_cortical_types.loc[economo_map, 'CorticalType'].astype('category').reset_index(drop=True)
    cortical_types_map = cortical_types_map.cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
    if parcellation_name:
        # load parcellation map
        parcellation_map = load_parcellation_map(parcellation_name, concatenate=True, downsampled=downsampled)
        parcellated_cortical_types_map = (
            #> create a dataframe of surface map including both cortical type and parcel index
            pd.DataFrame({'Cortical Type': cortical_types_map, 'Parcel': pd.Series(parcellation_map)})
            #> group by parcel
            .groupby('Parcel')
            #> find the cortical types with the highest count (may also be non-cortex)
            ['Cortical Type'].value_counts(sort=True).unstack().idxmax(axis=1)
            #> convert it back to category
            .astype('category')
            .cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
        )
        # assign cortical types of midline parcels to NaN
        parcellated_cortical_types_map.loc[helpers.MIDLINE_PARCELS[parcellation_name]] = np.NaN
        return parcellated_cortical_types_map
    else:
        return cortical_types_map      

def load_yeo_map(parcellation_name=None, downsampled=False):
    """
    Loads the map of Yeo networks

    Parameters
    ---------
    parcellation_name: (str or None)
        - None: vertex-wise
        - str: parcellation name (must exists in data/parcellation)
    downsampled: (bool) 
        load downsampled map of cortical types

    Returns
    --------
    yeo_map: (pd.Series:category)
        with the length n_vertices|n_parcels
    """
    yeo_maps = {}
    for hem in ['L', 'R']:
        yeo_giftii = nibabel.load(
            os.path.join(
                SRC_DIR,
                f'tpl-bigbrain_hemi-{hem}_desc-Yeo2011_7Networks_N1000.label.gii')
            )
        yeo_maps[hem] = yeo_giftii.darrays[0].data
        if downsampled:
            # select only the vertices corresponding to the downsampled ico5 surface
            mat = scipy.io.loadmat(
                os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
            )
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            yeo_maps[hem] = yeo_maps[hem][bb_downsample_indices]
    # concatenate the hemispheres and convert to pd category
    yeo_names = [
        'None', 'Visual', 'Somatomotor', 'Dorsal attention', 
        'Ventral attention', 'Limbic', 'Frontoparietal', 'Default'
        ]
    yeo_map = np.concatenate([yeo_maps['L'], yeo_maps['R']])
    yeo_map = pd.Series(yeo_map).astype('category').cat.rename_categories(yeo_names)
    if parcellation_name:
        # load parcellation map
        parcellation_map = load_parcellation_map(parcellation_name, concatenate=True, downsampled=downsampled)
        parcellated_yeo_map = (
            #> create a dataframe of surface map including both cortical type and parcel index
            pd.DataFrame({'Yeo Network': yeo_map, 'Parcel': pd.Series(parcellation_map)})
            #> group by parcel
            .groupby('Parcel')
            #> find the cortical types with the highest count (may also be non-cortex)
            ['Yeo Network'].value_counts(sort=True).unstack().idxmax(axis=1)
            #> convert it back to category
            .astype('category')
            .cat.reorder_categories(yeo_names)
        )
        # assign Yeo network of midline parcels to NaN
        parcellated_yeo_map.loc[helpers.MIDLINE_PARCELS[parcellation_name]] = np.NaN
        # TODO: this parcellation approach does not work perfectly
        # e.g. for schaefer 400 the 7Networks_LH_Cont_PFCl_1 parcel
        # is assigned to DMN! Maybe this is because of differences in Schaefer
        # versions (one is 400, and the other is 1000)
        # TODO: also sometimes the networks of the same parcel using
        # downsampled or original data is different (e.g. parcel 3 in SJH)
        # ideally don't use downsampled and parcellation_map functions together
        return parcellated_yeo_map
    else:
        return yeo_map


def load_exc_masks(exc_regions, concatenate=False, downsampled=False):
    """
    Create masks of bigbrain space including agranular and dysgranular region.

    Parameters
    ---------
    exc_regions: (str | None)
        - allocortex: excludes allocortex
        - adysgranular: excludes allocortex + adysgranular regions

    concatenate: (bool)

    Returns
    -------
    exc_mask: (np.ndarray | dict) boolean surface maps of exclusion mask
    """
    # load cortical types surface map after/without parcellation
    cortical_types = load_cortical_types(downsampled=downsampled)
    # # if it's parcellated project back to surface
    # #  Why? because parcellation_map and economo parcellation do
    # #  not fully overlap and we need a mask that is aligned with the parcellation_map
    # if parcellation_name: 
    #     cortical_types = pd.Series(helpers.deparcellate(cortical_types, parcellation_name).flatten())
    # # match midlines of the `parcellation_name` and economo parcellation (sometimes e.g. in sjh
    # #  they do not match) by makign their overlap NaN. The midline of `parcellation_name` is already
    # #  NaN (from load_cortical_types). Therefore only load midline of economo and make it NaN
    # economo_map = load_parcellation_map('economo', concatenate=True)
    # economo_midline_mask = np.in1d(economo_map, ['unknown', 'corpuscallosum'])
    # cortical_types[economo_midline_mask] = np.NaN
    # create a mask of excluded cortical types
    if exc_regions == 'allocortex':
        exc_cortical_types = [np.NaN, 'ALO']
    elif exc_regions == 'adysgranular':
        exc_cortical_types = [np.NaN, 'ALO', 'DG', 'AG']
    exc_mask = cortical_types.isin(exc_cortical_types).values
    if concatenate:
        return exc_mask
    else:
        return {
            'L': exc_mask[:N_VERTICES_HEM_BB],
            'R': exc_mask[N_VERTICES_HEM_BB:]
        }

def load_laminar_thickness(exc_masks=None, normalize_by_total_thickness=True, 
                           regress_out_curvature=False, smooth_disc_radius=None):
    """
    Loads laminar thickness data from 'data' folder and after masking out
    `exc_mask` returns 6-d laminar thickness arrays for left and right hemispheres.
    Also does normalization, regressing out of the curvature or disc smoothing
    if indicated.

    Parameters
    --------
    exc_masks: (dict of np.ndarray) 
        The surface masks of vertices that should be excluded (L and R)
    normalize_by_total_thickness: (bool) 
        Normalize by total thickness. Default: True
    regress_out_curvature: (bool) 
        Regress out curvature. Default: False
    smooth_disc_radius: (int | None) 
        Smooth the absolute thickness of each layer using a disc with the given radius.

    Retruns
    --------
    laminar_thickness: (dict of np.ndarray) n_vertices [ico7 or ico5] x 6 for laminar thickness of L and R hemispheres
    """
    laminar_thickness = {}
    for hem in ['L', 'R']:
        # read the laminar thickness data from bigbrainwrap .txt files
        laminar_thickness[hem] = np.empty((N_VERTICES_HEM_BB, 6))
        for layer_num in range(1, 7):
            laminar_thickness[hem][:, layer_num-1] = np.loadtxt(
                os.path.join(
                    SRC_DIR,
                    f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
                    ))
        # remove the exc_mask
        if exc_masks:
            laminar_thickness[hem][exc_masks[hem], :] = np.NaN
    if smooth_disc_radius:
        # downsample it for better performance
        laminar_thickness = helpers.downsample(laminar_thickness)
        # smooth the laminar thickness using the disc approach
        laminar_thickness, _ = helpers.disc_smooth(laminar_thickness, smooth_disc_radius)
    for hem in ['L', 'R']:
        # normalize by total thickness
        if normalize_by_total_thickness:
            laminar_thickness[hem] /= laminar_thickness[hem].sum(axis=1, keepdims=True)
        # regress out curvature
        if regress_out_curvature:
            if smooth_disc_radius:
                logging.warning("Skipping regressing out of the curvature as laminar thickness is smoothed")
            else:
                cov_surf_data = load_curvature_maps()[hem]
                laminar_thickness[hem] = helpers.regress_out_surf_covariates(
                    laminar_thickness[hem], cov_surf_data,
                    sig_only=False, renormalize=True
                    )
    return laminar_thickness

def load_economo_laminar_thickness(normalize_by_total_thickness=True):
    """
    Loads laminar thickness data from von Economo atlas in von Economo
    parcellation

    Parameters
    --------
    normalize_by_total_thickness: (bool) 
        Normalize by total thickness. Default: True

    Retruns
    --------
    parcellated_laminar_thickness: (pd.DataFrame) n_parc x 6 layers
    """
    parcellated_laminar_thickness = pd.read_csv(
        os.path.join(
            SRC_DIR, 
            'von_economo_laminar_thickness.csv'
            ), 
        delimiter=";").set_index('area_name')
    if normalize_by_total_thickness:
        parcellated_laminar_thickness = parcellated_laminar_thickness.divide(
            parcellated_laminar_thickness.sum(axis=1),
            axis=0
            )
    return parcellated_laminar_thickness

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
    # load profiles and reshape to n_vert x 50
    density_profiles = np.loadtxt(os.path.join(SRC_DIR, 'tpl-bigbrain_desc-profiles.txt'))
    density_profiles = density_profiles.T
    # split hemispheres
    density_profiles = {
        'L': density_profiles[:density_profiles.shape[0]//2, :],
        'R': density_profiles[density_profiles.shape[0]//2:, :],
    }
    # remove the exc_mask
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
        # read the laminar thickness data from bigbrainwrap .txt files
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
        # remove the exc_mask
        if exc_masks:
            laminar_density[hem][exc_masks[hem], :] = np.NaN
        # TODO: also normalize density?
    return laminar_density

def load_hcp1200_myelin_map(exc_regions=None, downsampled=False):
    """
    Load HCP 1200 myelination map

    Parameters
    ---------
    exc_regions: (str | None)
    downsampled: (bool)
    """
    myelin_map = np.concatenate([
        nilearn.surface.load_surf_data(
            os.path.join(
                SRC_DIR, 
                'tpl-bigbrain_hemi-L_desc-hcp1200_myelinmap.shape.gii'
                )),
        nilearn.surface.load_surf_data(
            os.path.join(
                SRC_DIR, 
                'tpl-bigbrain_hemi-R_desc-hcp1200_myelinmap.shape.gii'
                ))
    ])
    if downsampled:
        myelin_map = helpers.downsample(myelin_map)
    if exc_regions is None:
        # get a maks of midline (from sjh parcellation)
        exc_mask = np.isnan(
            helpers.deparcellate(
                helpers.parcellate(myelin_map, 'sjh'), 
                'sjh', 
                downsampled=downsampled
                )[:, 0])
    else:
        exc_mask = load_exc_masks(
            exc_regions, 
            concatenate=True, 
            downsampled=downsampled
            )
    myelin_map[exc_mask] = np.NaN
    return myelin_map


def load_parcellation_map(parcellation_name, concatenate, downsampled=False, load_indices=False):
    """
    Loads parcellation maps of L and R hemispheres, correctly relabels them
    and concatenates them if `concatenate` is True

    Parameters
    ----------
    parcellation_name: (str) Parcellation scheme
    concatenate: (bool) cocnatenate the hemispheres
    downsampled: (bool) load a downsampled version of parcellation map
    load_indices: (bool) return the parcel indices instead of their names

    Returns
    -------
    parcellation_map: (np.ndarray or dict of np.ndarray)
    """
    parcellation_map = {}
    for hem in ['L', 'R']:
        # load parcellation map
        parcellation_map[hem] = nilearn.surface.load_surf_data(
            os.path.join(
                SRC_DIR, 
                f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation.label.gii')
            )
        if not load_indices:
            # label parcellation map
            _, _, sorted_labels = nibabel.freesurfer.io.read_annot(
                os.path.join(
                    SRC_DIR, 
                    f'{hem.lower()}h_{parcellation_name}.annot')
            )
            # labels post-processing for each specific parcellation
            if parcellation_name == 'sjh':
                # remove sjh_ from the label
                sorted_labels = list(map(lambda l: int(l.decode().replace('sjh_','')), sorted_labels))
            elif parcellation_name in ['aparc', 'economo']:
                # add hemisphere to the labels to have distinct parcel labels in each hemisphere
                sorted_labels = list(map(lambda l: f'{hem}_{l.decode()}', sorted_labels))
            else:
                sorted_labels = list(map(lambda l: l.decode(), sorted_labels)) # b'name' => 'name'
            transdict = dict(enumerate(sorted_labels))
            parcellation_map[hem] = np.vectorize(transdict.get)(parcellation_map[hem])
        if downsampled:
            # select only the vertices corresponding to the downsampled ico5 surface
            # Warning: For fine grained parcels such as sjh this approach leads to
            # a few parcels being removed
            mat = scipy.io.loadmat(
                os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                )
            )
            bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
            parcellation_map[hem] = parcellation_map[hem][bb_downsample_indices]
    if concatenate:
        return np.concatenate([parcellation_map['L'], parcellation_map['R']])
    else:
        return parcellation_map

def load_volumetric_parcel_labels(parcellation_name):
    """
    Loads the lables corresponding to the parcels in volumetric
    space, except for the parcel ID 0, which corresponds to
    background / midline

    Parameter
    --------
    parcellation_name: (str)
    """
    if 'schaefer' in parcellation_name:
        # For schaefer load the names from color tables
        n_parcels = int(parcellation_name.replace('schaefer', ''))
        url = ('https://github.com/ThomasYeoLab/CBIG/raw/master/'
               'stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/'
               'Parcellations/MNI/freeview_lut/'
               f'Schaefer2018_{n_parcels}Parcels_7Networks_order.txt')
        labels = pd.read_csv(url, sep='\t', header=None)[1].values
    elif parcellation_name == 'sjh':
        # For SJH the volumetric parcel values are already
        # parcel labels (i.e., 'sjh_{id}'), which also corresponds
        # to the parcel ids from the surface, and simply getting
        # the unique ids from surface parcellation would be the
        # correct labels
        labels = np.unique(load_parcellation_map('sjh', True))
        labels = np.delete(labels, 0)
    elif parcellation_name == 'aparc':
        # Get the lables from abagen desikan killiany atlas and remove
        # subcortical lables
        dk_info = pd.read_csv(abagen.fetch_desikan_killiany()['info'])
        dk_labels = dk_info['hemisphere'] + '_' + dk_info['label']
        labels = dk_labels[dk_info['structure']=='cortex'].values
    return labels

def load_disease_maps(psych_only, rename=False):
    """
    Loads maps of cortical thickness difference in disorders.
    Adult popultation, more general categories
    of disorders (e.g. all epliepsy vs TLE) and meta-analyses
    (over mega-analyses) are preferred

    Parameters
    ---------
    psych_only: (bool)
        only include psychiatric disorders
    rename: (bool)
        rename the disorders for publicationß

    Returns
    -------
    parcellated_disorder_maps: (pd.DataFrame)
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
        ['CortThick_case_vs_controls_meta_analysis']
        .set_index('Structure')['d_icv'])
    if not psych_only:
        parcellated_disorder_maps['epilepsy'] = (
            enigmatoolbox.datasets.load_summary_stats('epilepsy')
            ['CortThick_case_vs_controls_allepilepsy']
            .set_index('Structure')['d_icv'])
    if rename:
        parcellated_disorder_maps = parcellated_disorder_maps.rename(
            columns = {
                'adhd': 'ADHD',
                'bipolar': 'BD',
                'depression': 'MDD',
                'ocd': 'OCD',
                'schizophrenia': 'SCZ',
                'asd': 'ASD',
                'epilepsy': 'Epilepsy'
            }
        )
    return parcellated_disorder_maps


def load_conn_matrix(kind, parcellation_name='schaefer400', dataset='hcp'):
    """
    Loads FC or SC matrices in Schaefer parcellation (400) from ENIGMA toolbox
    and reorders it according to `matrix_file`. For SC matrix also makes contralateral
    values 0 (so that they are not included in correlations)

    Parameters
    ----------
    kind: (str)
        - structural
        - functional
        - effective
    parcellation_name: (str)
        - schaefer400 (currently only this is supported)
    dataset: (str) for the effective connectivity
        - hcp
        - mics

    Returns
    ---------
    reordered_conn_matrix, (np.ndarray) (n_parc, n_parc) 
        reordered FC or SC matrices matching the original matrix
    """
    if kind == 'effective':
        # load the rDCM results from Paquola 2021
        results = scipy.io.loadmat(
            os.path.join(SRC_DIR, f'{dataset}_rDCM_sch400.mat'), 
            mat_dtype=True, struct_as_record=False)['results'][0][0]
        conn_matrix = results.mean_Amatrix_allSubjects
        # get the absolute values and zero out the diagonal
        conn_matrix = np.abs(conn_matrix)
        conn_matrix[np.diag_indices_from(conn_matrix)] = 0
        conn_matrix_labels = [a[0] for a in results.AllRegions[0]]
    else:
        # match parcellation name with the enigmatoolbox
        ENIGMATOOLBOX_PARC_NAMES = {
            'schaefer400': 'schaefer_400'
        }
        enigma_parcellation_name = ENIGMATOOLBOX_PARC_NAMES.get(parcellation_name, None)
        if enigma_parcellation_name == None:
            raise Exception(f"ENIGMA Toolbox does not have connectivity data for the parcellation {parcellation_name}")
        # load data from enigma toolbox
        if kind == 'structural':
            conn_matrix, conn_matrix_labels, _, _ = enigmatoolbox.datasets.load_sc(enigma_parcellation_name)
        else:
            conn_matrix, conn_matrix_labels, _, _ = enigmatoolbox.datasets.load_fc(enigma_parcellation_name)
    conn_matrix = pd.DataFrame(conn_matrix, columns=conn_matrix_labels, index=conn_matrix_labels)
    # parcellate dummy data to get the order of parcels in other matrices created locally
    dummy_surf_data = np.zeros(N_VERTICES_HEM_BB * 2)
    parcellated_dummy = helpers.parcellate(dummy_surf_data, parcellation_name).dropna()
    # reorder matrices downloaded from enigmaltoolbox
    reordered_conn_matrix = conn_matrix.loc[parcellated_dummy.index, parcellated_dummy.index]
    return reordered_conn_matrix

def fetch_ahba_data(parcellation_name, return_donors=False, 
                    discard_rh=True, **abagen_kwargs):
    """
    Gets the parcellated AHBA gene expression data using abagen. The
    data is either an aggregated dataframe across all donors or is
    a dictionary of dataframes with the data for each donor. The latter
    is not be normalized across the regions as this normalization
    will be done after aggregate gene expressions (e.g. for neuronal subtypes)
    are calculated.

    Parameters
    ---------
    parcellation_name: (str)
    return_donors: (bool)
    discard_rh: (bool)
    and other abagen.get_expression_data keyword arguments

    Returns
    -------
    ahba_data: (pd.DataFrame | dict) 
    """
    # TODO: this doesn't work with aparc parcellation
    # specify the file path and load it if it exists
    file_path = os.path.join(
        SRC_DIR,
        'ahba'\
        + ('_donors' if return_donors else '')\
        + f'_parc-{parcellation_name}'\
        + ('_hemi-L' if discard_rh else '')\
        + ''.join((f'_{k}-{v}' for k, v in abagen_kwargs.items()))\
        + '.npz'
    )
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)['data'].tolist()
    # otherwise load the data from abagen
    # specify the config
    if os.path.exists(ABAGEN_DIR):
        data_dir = ABAGEN_DIR
    else:
        data_dir = None
    atlas = (os.path.join(SRC_DIR, f'lh_{parcellation_name}_fsa5.label.gii'),
            os.path.join(SRC_DIR, f'rh_{parcellation_name}_fsa5.label.gii'))
    if return_donors:
        # avoid normalizing across samples before aggregate for subtypes are calculated
        gene_norm = None
    else:
        gene_norm = 'srs'
    # get the data as dict
    expression_data = abagen.get_expression_data(
        atlas = atlas,
        data_dir = data_dir,
        gene_norm = gene_norm, 
        return_donors = return_donors,
        verbose = 2,
        **abagen_kwargs
    )
    if not return_donors:
        expression_data = {'all': expression_data}
    # replace the index with parcel labels and remove the
    # midline and rename the parcels similar to load_parcellation_map if needed
    _, atlas_info = abagen.images.check_surface(atlas)
    if parcellation_name == 'sjh':
        # remove sjh_ from the lable
        atlas_info['label'] = atlas_info['label'].map(lambda l: int(l.replace('sjh_','')))
    parcels_mask = ~(atlas_info['label'].isin(helpers.MIDLINE_PARCELS['sjh']))
    if discard_rh:
        parcels_mask = parcels_mask & (atlas_info['hemisphere']=='L')
    for donor in expression_data.keys():
        expression_data[donor] = expression_data[donor].loc[parcels_mask]
        expression_data[donor].index = atlas_info.loc[parcels_mask, 'label']
    # save
    np.savez_compressed(file_path, 
                        data=expression_data)
    return expression_data

def fetch_aggregate_gene_expression(gene_list, parcellation_name, discard_rh=True,
                                    merge_donors='genes', avg_method='mean', missing='centroids',
                                    ibf_threshold = 0.25, **abagen_kwargs):
    """
    Gets the aggregate expression of genes in `gene_list`

    Parameters
    ---------
    gene_list: (list | pd.Series)
        list of gene names or a series with names in the index
        and weights in the values
        Note: This will ignore genes that do not exist in the current
        version of gene expression data
    parcellation_name: (str)
    discard_rh: (bool)
        limit the map to the left hemisphere
            Note: For consistency with other functions the right
            hemisphere vertices/parcels are not removed but are set
            to NaN
    merge_donors: (str | None)
        - genes: merge donors at the level of individual genes (done in abagen)
        - aggregates: merge donors after calculating aggregates separately in each donor
        - None: do not merge donors
    avg_method: (str)
        - mean
        - median (ignores the weights in gene list)
    missing: (str) 
        how to deal with parcels missing data
            - None
            - centroids
            - interpolate
    ibf_threshold: (float)
    """
    # get the ahba data
    # do not return donor-specific data if 
    # merging of donors should be done at
    # genes level
    ahba_data = fetch_ahba_data(
        parcellation_name, 
        return_donors = (merge_donors!='genes'),
        discard_rh = discard_rh,
        ibf_threshold = ibf_threshold,
        missing = missing,
        **abagen_kwargs
        )
    # get the gene list that exist in ahba data
    ahba_genes = list(ahba_data.values())[0].columns
    if isinstance(gene_list, list):
        # if no weights are given set the weight
        # of all genes to 1
        gene_list = pd.Series(1, index=gene_list)
    exist_gene_list = (set(gene_list.index) & set(ahba_genes))
    logging.info(f'{gene_list.shape[0] - len(exist_gene_list)} of {gene_list.shape[0]} genes do not exist')
    gene_list = gene_list.loc[exist_gene_list]
    # get the aggregate expression for each donor
    # (in the case of return_donors==False there's
    # only one donor named 'all')
    aggregate_expressions = {}
    for donor, donor_df in ahba_data.items():
        if avg_method == 'mean':
            aggregate_expressions[donor] = (
                (
                    donor_df.loc[:, gene_list.index] # gene expression
                    @ gene_list.values # weights
                )
                / gene_list.sum()
            )
        elif avg_method == 'median':
            aggregate_expressions[donor] = donor_df.loc[:, gene_list.index].median(axis=1)
        if merge_donors!='genes':
            # normalize the aggregate of genes if donor-specific
            # expression are used, as the individual genes are 
            # not already normalized. For the data that is already
            # aggregated (donor == 'all') the normalization is already
            # done at the level of individual genes
            valid_parcels = aggregate_expressions[donor].dropna().index
            aggregate_expressions[donor].loc[valid_parcels] = \
                abagen.correct._srs(
                    aggregate_expressions[donor].loc[valid_parcels]
                    )
    if merge_donors is not None:
        return pd.DataFrame(
            np.nanmean(np.array(list(aggregate_expressions.values())), axis=0),
            index=aggregate_expressions[donor].index
        )
    else:
        return aggregate_expressions

def fetch_pet(parcellation_name, receptor):
    """
    Loads the parcellated PET map of receptor and Z-scores 
    the maps and takes a weighted average in case multiple 
    maps exist for a given receptor x tracer combination

    Parameters
    ----------
    parcellation_name: (str)
    receptor: (str)

    Returns
    ---------
    parcellated_data: (pd.DataFrame) n_parcels x n_receptor_tracer_groups
    """
    # TODO: consider loading the data online from neuromaps
    parcellated_data = pd.DataFrame()
    # load PET images metadata
    metadata = pd.read_csv(
        os.path.join(SRC_DIR, 'PET_metadata.csv'), 
        index_col='filename')
    metadata = metadata.loc[metadata['receptor']==receptor]
    # group the images with the same recetpro-tracer
    for group, group_df in metadata.groupby(['receptor', 'tracer']):
        group_name = '_'.join(group)
        logging.info(group_name)
        # take a weighted average of PET value z-scores
        # across images with the same receptor-tracer
        # (weighted by N of subjects)
        pet_parcellated_sum = {}
        for filename, file_metadata in group_df.iterrows():
            pet_img = os.path.join(SRC_DIR, 'PET', filename)
            #> prepare the parcellation masker
            # Warning: Background label is by default set to
            # 0. Make sure this is the case for all the parcellation
            # maps and zero corresponds to background / midline
            masker = NiftiLabelsMasker(
                os.path.join(
                    SRC_DIR, 
                    f'tpl-MNI152_desc-{parcellation_name}_parcellation.nii.gz'
                    ), 
                strategy='sum',
                resampling_target='data',
                background_label=0)
            #> count the number of non-zero voxels per parcel so the average
            # is calculated only among non-zero voxels (visualizing the PET map
            # on volumetric parcellations, the parcels are usually much thicker
            # than the PET map on the cortex, and there are a large number of 
            # zero PET values in each parcel which can bias the parcelled values)
            nonzero_mask = math_img('pet_img != 0', pet_img=pet_img)
            nonzero_voxels_count_per_parcel = masker.fit_transform(nonzero_mask).flatten()
            #> take the average of PET values across non-zero voxels
            pet_value_sum_per_parcel = masker.fit_transform(pet_img).flatten()
            pet_parcellated = pet_value_sum_per_parcel / nonzero_voxels_count_per_parcel
            # TODO: should I make any transformations in the negative PET images?
            #> get the PET intensity zscore weighted by N
            pet_parcellated_sum[filename] = (
                scipy.stats.zscore(pet_parcellated)
                * file_metadata['N']
            )
        # divide the sum of weighted Z-scores by total N
        # Note that in the case of one file per group we can avoid
        # multiplying by N and dividing by sum of N, but I've
        # used this approach to have a shorter code which can
        # also support the option of merging by receptor
        # and not only (receptor, tracer) combinations
        parcellated_data.loc[:, group_name] = sum(pet_parcellated_sum.values()) / group_df['N'].sum()
        # add labels of the parcels
        parcellated_data.index = load_volumetric_parcel_labels(parcellation_name)
    return parcellated_data


def fetch_autoradiography():
    """
    Loads the parcellated map of receptors based on autoradiography
    
    Credit
    ------
    Original data from Zilles 2017 (10.3389/fnana.2017.00078)
    Saved in .npy format by Goulas 2021 (10.1073/pnas.2020574118)
    Code and mapping for conversion from JuBrain/Brodmann to DK from
    Hansen 2021 (https://www.biorxiv.org/content/10.1101/2021.11.30.469876v2;
    https://github.com/netneurolab/hansen_gene-receptor/blob/main/code/main.py)
    """
    # load the receptor regions and create a mapping between
    # them and DK parcels
    receptor_regions = np.load(
        os.path.join(SRC_DIR, 'autoradiography', 'RegionNames.npy')
    )
    receptor_regions_idx = np.arange(receptor_regions.size)
    # some region indices are associated with more than one dk region
    # the data for these regions should be duplicated
    duplicate = [20, 21, 28, 29, 30, 32, 34, 39]
    rep = np.ones((receptor_regions.shape[0], ), dtype=int) 
    rep[duplicate] = 2
    receptor_regions = np.repeat(receptor_regions, rep, 0)
    receptor_regions_idx = np.repeat(receptor_regions_idx, rep, 0)
    # mapping from 44 receptor regions + 7 duplicate regions to dk left hem
    # manually done (by Hansen et al.), comparing anatomical regions to one another
    mapping = np.array([57, 57, 57, 57, 63, 62, 65, 62, 64, 65, 64, 66, 66,
                        66, 66, 66, 74, 74, 70, 71, 72, 73, 67, 68, 69, 52,
                        52, 60, 60, 58, 58, 59, 53, 54, 53, 54, 55, 56, 61,
                        51, 51, 50, 49, 49, 44, 44, 45, 42, 47, 46, 48, 43])
    # get the labels of DK regions (== scale033 of Cammoun2012)
    # based on IDs in `mapping` and create a dataframe for
    # mapping between DK region labels and receptor regions
    cammoun = netneurotools.datasets.fetch_cammoun2012()
    info = pd.read_csv(cammoun['info'])
    mapping_df = info[(info['scale']=='scale033')].set_index('id').loc[mapping, 'label'].to_frame()
    ## make the lables consistent with the way load_parcellation_map renames DK parcels
    mapping_df.loc[:, 'label'] = 'L_' + mapping_df.loc[:, 'label']
    ## add receptor region label and indices corresponding to the DK parcels
    ## as indicated in `mapping`
    mapping_df.loc[:,'receptor_region'] = receptor_regions
    mapping_df.loc[:,'receptor_region_idx'] = receptor_regions_idx
    mapping_df = mapping_df.set_index('receptor_region_idx')
    # load the original data and take average across 
    # cortical depth before any normalization
    autorad_data_orig = []
    for depth in ['S', 'G', 'I']: # supragranular, granular, infragranular
        autorad_data_orig.append(
            np.load(os.path.join(
                SRC_DIR, 'autoradiography', f'ReceptData_{depth}.npy' 
            ))
        )
    autorad_data_orig = sum(autorad_data_orig) / 3
    receptor_names = np.load(
        os.path.join(SRC_DIR, 'autoradiography', 'ReceptorNames.npy')
        ).tolist()
    # create a dataframe of autoradiography data for DK regions
    # based on the mapping between receptor data regions and DK regions
    autorad_data = (
        pd.DataFrame(
            ## duplicate receptor data in receptor regions covering
            ## multiple DK regions
            autorad_data_orig[mapping_df.index],
            index=mapping_df['label'],
            columns=receptor_names)
        ## take the average data for DK regions covering multiple
        ## receptor regions
        .reset_index(drop=False).groupby('label').mean()
        ## normalize across regions
        .apply(scipy.stats.zscore, axis=0)
    )
    return autorad_data
