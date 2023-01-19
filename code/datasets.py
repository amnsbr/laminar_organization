import os
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
import neuromaps
import re


# specify the directories and constants
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
ABAGEN_DIR = '/data/group/cng/abagen-data'
N_VERTICES_HEM_BB = 163842
N_VERTICES_HEM_BB_ICO5 = 10242
N_VERTICES_HEM_FSLR = 32492
LAYERS_COLORS = {
    'bigbrain': ['#abab6b', '#dabcbc', '#dfcbba', '#e1dec5', '#66a6a6','#d6c2e3'], 
    'wagstyl': ['#3a6aa6ff', '#f8f198ff', '#f9bf87ff', '#beaed3ff', '#7fc47cff','#e31879ff']
}

def load_mesh_paths(kind='orig', space='bigbrain', downsampled=True):
    """
    Loads or creates surfaces of bigbrain/fsa left and right hemispheres
    For bigbrain downsampled the actual downsampling is done in 
    `local/downsample_bb.m` in matlab and this function just makes 
    it python-readable

    Parameters
    ----------
    kind: (str)
        - 'orig'
        - 'inflated'
        - 'sphere'
    space: (str)
        - 'bigbrain'
        - 'fsaverage'
        - 'fs_LR'
        - 'yerkes' (macaque)
    downsampled: (bool)

    Returns
    ------
    paths: (dict of str) path to downsampled surfaces of L and R
    """
    # TODO: add sphere and inflated for the other spaces
    paths = {}
    for hem in ['L', 'R']:
        if space=='bigbrain':
            if downsampled:
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
                # save it as gifti by using ico7 gifti as a template
                # and modifying it
                bb10_gifti = nibabel.load(
                    os.path.join(
                        SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.inflate.gii'
                    )
                )
                bb10_gifti.darrays[0].data = coords
                bb10_gifti.darrays[0].dims = list(coords.shape)
                bb10_gifti.darrays[1].data = faces
                bb10_gifti.darrays[1].dims = list(faces.shape)
                nibabel.save(bb10_gifti, paths[hem])
            else:
                if kind=='orig':
                    paths[hem] = os.path.join(
                        SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                    )
                elif kind=='inflated':
                    paths[hem] = os.path.join(
                        SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.inflate.gii'
                    )
        elif space == 'fsaverage':
            hem_fullname = {'L':'left', 'R':'right'}
            if downsampled:
                fsa_version = 'fsaverage5'
            else:
                fsa_version = 'fsaverage'
            if kind=='orig':
                paths[hem] = nilearn.datasets.fetch_surf_fsaverage(fsa_version)[f'pial_{hem_fullname[hem]}']
            elif kind=='inflated':
                paths[hem] = nilearn.datasets.fetch_surf_fsaverage(fsa_version)[f'infl_{hem_fullname[hem]}']      
        elif space == 'fs_LR':
            if kind == 'orig':
                paths[hem] = os.path.join(SRC_DIR, f'S1200.{hem}.pial_MSMAll.32k_fs_LR.surf.gii')
            else:
                raise NotImplementedError
        elif space == 'yerkes':
            if kind == 'orig':
                paths[hem] = os.path.join(SRC_DIR, f'MacaqueYerkes19.{hem}.pial.32k_fs_LR.surf.gii')
            elif kind == 'inflated':
                paths[hem] = os.path.join(SRC_DIR, f'MacaqueYerkes19.{hem}.inflated.32k_fs_LR.surf.gii')
            elif kind == 'sphere':
                paths[hem] = os.path.join(SRC_DIR, f'MacaqueYerkes19.{hem}.sphere.32k_fs_LR.surf.gii')
    return paths

def load_curvature_maps(downsampled=False, concatenate=False):
    """
    Creates the map of curvature for the bigbrain surface
    using pycortex or loads it if it already exist

    Parameters
    ---------
    downsampled: (bool)
    concatenate: (bool)

    Returns
    -------
    curvature_maps: (np.ndarray | dict) n_vertices, for L and R hemispheres
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
        mesh_path = load_mesh_paths('orig', downsampled=downsampled)[hem]
        vertices, faces = nilearn.surface.load_surf_mesh(mesh_path)
        surface = Surface(vertices, faces)
        # calculate mean curvature; negative = sulci, positive = gyri
        curvature = surface.mean_curvature()
        # save it
        os.makedirs(os.path.join(OUTPUT_DIR, 'curvature'), exist_ok=True)
        np.save(curvature_filepath, curvature)
        curvature_maps[hem] = curvature
    if concatenate:
        return np.concatenate([curvature_maps['L'], curvature_maps['R']])
    else:
        return curvature_maps

def load_cortical_types(parcellation_name=None, space='bigbrain', downsampled=False):
    """
    Loads the map of cortical types

    Parameters
    ---------
    parcellation_name: (str or None)
    sapce: (str)
    downsampled: (bool)

    Returns
    --------
    cortical_types_map (pd.DataFrame): with the length n_vertices|n_parcels
    """
    # load the economo map
    economo_map = load_parcellation_map('economo', True, downsampled=downsampled, space=space)
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
        # TODO: may return different types for some parcels depending on wether the surface should
        # be downsampled (e.g. parcel 315 of sjh)
        parcellation_map = load_parcellation_map(parcellation_name, concatenate=True, space=space, downsampled=downsampled)
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
        parcellated_cortical_types_map.loc[helpers.MIDLINE_PARCELS.get(parcellation_name,[])] = np.NaN
        return parcellated_cortical_types_map
    else:
        return cortical_types_map      

def load_yeo_map(parcellation_name=None, downsampled=False):
    """
    Loads the map of Yeo networks

    Parameters
    ---------
    parcellation_name: (str or None)
    downsampled: (bool) 

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
    # specify list of excluded types
    if exc_regions == 'allocortex':
        exc_cortical_types = [np.NaN, 'ALO']
    elif exc_regions == 'adysgranular':
        exc_cortical_types = [np.NaN, 'ALO', 'DG', 'AG']
    # create mask +/- concatenate
    exc_mask = cortical_types.isin(exc_cortical_types).values
    if concatenate:
        return exc_mask
    else:
        return {
            'L': exc_mask[:N_VERTICES_HEM_BB],
            'R': exc_mask[N_VERTICES_HEM_BB:]
        }

def load_laminar_thickness(exc_regions=None, normalize_by_total_thickness=True, 
        regress_out_curvature=False, smooth_disc_radius=None, smooth_disc_approach='euclidean'):
    """
    Loads BigBrain laminar thickness data, excludes `exc_regions`, performs 
    smoothing or regression of curvature to correct for it, and normalizes by
    total thickness.

    Parameters
    --------
    exc_regions: (str | None) 
    normalize_by_total_thickness: (bool) 
    regress_out_curvature: (bool) 
    smooth_disc_radius: (int | None) 
        Smooth the absolute thickness of each layer using a disc with the given radius.
    smooth_disc_approach: (str)
        - euclidean
        - geodesic

    Returns
    --------
    laminar_thickness: (dict of np.ndarray) n_vertices [ico7 or ico5] x 6 for laminar thickness of L and R hemispheres
    """
    if exc_regions is not None:
        exc_masks = load_exc_masks(exc_regions)
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
        if exc_regions is not None:
            laminar_thickness[hem][exc_masks[hem], :] = np.NaN
    if smooth_disc_radius:
        # smoothing is memory intensive, load it from disc
        # if it is created already
        smoothed_laminar_thickness_path = os.path.join(
            SRC_DIR, 
            f'tpl-bigbrain_desc-laminar_thickness_exc-{exc_regions}_smooth-{smooth_disc_radius}_approach-{smooth_disc_approach}.npz'
        )
        if os.path.exists(smoothed_laminar_thickness_path):
            laminar_thickness = np.load(smoothed_laminar_thickness_path)
            # convert npz object to dict (even though they have the same indexing)
            laminar_thickness = {
                'L': laminar_thickness['L'],
                'R': laminar_thickness['R']
            }
        else:
            # downsample it for better performance
            laminar_thickness = helpers.downsample(laminar_thickness)
            # smooth the laminar thickness using the disc approach
            laminar_thickness, _ = helpers.disc_smooth(laminar_thickness, smooth_disc_radius, smooth_disc_approach)
            # save it
            np.savez_compressed(
                smoothed_laminar_thickness_path,
                L = laminar_thickness['L'],
                R = laminar_thickness['R']
            )
    for hem in ['L', 'R']:
        # normalize by total thickness
        if normalize_by_total_thickness:
            laminar_thickness[hem] /= laminar_thickness[hem].sum(axis=1, keepdims=True)
        # regress out curvature
        if regress_out_curvature:
            if smooth_disc_radius:
                print("Skipping regressing out of the curvature as laminar thickness is smoothed")
            else:
                cov_surf_data = load_curvature_maps()[hem]
                laminar_thickness[hem] = helpers.regress_out_surf_covariates(
                    laminar_thickness[hem], cov_surf_data,
                    sig_only=False, renormalize=True
                    )
    return laminar_thickness

def load_total_depth_density(exc_regions=None):
    """
    Loads laminar density of total cortical depth sampled at 50 points and after masking out
    `exc_regions` returns separate arrays for L and R hemispheres

    Parameters
    ---------
    exc_regions: (str | None) The surface masks of vertices that should be excluded (L and R)
 

    Returns
    -------
    density_profiles (dict of np.ndarray) 
        n_vert x 50 for L and R hemispheres
    """
    if exc_regions is not None:
        exc_masks = load_exc_masks(exc_regions)
    # load profiles and reshape to n_vert x 50
    density_profiles = np.loadtxt(os.path.join(SRC_DIR, 'tpl-bigbrain_desc-profiles.txt'))
    density_profiles = density_profiles.T
    # split hemispheres
    density_profiles = {
        'L': density_profiles[:density_profiles.shape[0]//2, :],
        'R': density_profiles[density_profiles.shape[0]//2:, :],
    }
    # remove the exc_mask
    if exc_regions is not None:
        for hem in ['L', 'R']:
            density_profiles[hem][exc_masks[hem], :] = np.NaN
    return density_profiles


def load_laminar_density(exc_regions=None, method='mean'):
    """
    Loads BigBrain laminar density data, takes the average of sample densities
    for each layer, and after masking out `exc_mask` returns 6-d average laminar density 
    arrays for left and right hemispheres.
    The laminar density data is created by `local/create_laminar_density_profiles.sh`

    Parameters
    ----------
    exc_regions: (str | None)
    method: (str) method of finding central tendency of samples in each layer in each vertex
        - mean
        - median

    Returns
    --------
    laminar_density: (dict of np.ndarray) 
        n_vertices x 6 for laminar density of L and R hemispheres
    """
    if exc_regions is not None:
        exc_masks = load_exc_masks(exc_regions)
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
        if exc_regions is not None:
            laminar_density[hem][exc_masks[hem], :] = np.NaN
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


def load_parcellation_map(parcellation_name, concatenate, downsampled=False, 
        load_indices=False, space='bigbrain'):
    """
    Loads parcellation maps of L and R hemispheres, correctly relabels them
    and concatenates them if `concatenate` is True

    Parameters
    ----------
    parcellation_name: (str) Parcellation scheme
    concatenate: (bool) cocnatenate the hemispheres
    downsampled: (bool) load a downsampled version of parcellation map
    load_indices: (bool) return the parcel indices instead of their names
    space: (str)
        - 'bigbrain' (default)
        - 'fsaverage'

    Returns
    -------
    parcellation_map: (np.ndarray or dict of np.ndarray)
    """
    if space == 'yerkes':
        yerkes_cifti = nibabel.load(os.path.join(SRC_DIR, 'Yerkes19_Parcellations_v2.32k_fs_LR.dlabel.nii'))
        if parcellation_name == 'M132':
            concat_parcellation_map = yerkes_cifti.get_fdata()[5, :]
            parcellation_map = {
                'L': concat_parcellation_map[:N_VERTICES_HEM_FSLR],
                'R': concat_parcellation_map[N_VERTICES_HEM_FSLR:],
            }
            if not load_indices:
                raw_labels = yerkes_cifti.header.get_axis(0).label[5]
                for hem in ['L', 'R']:
                    labels = [hem+'_'+l[0].replace('_M132', '') for l in raw_labels.values()]
                    transdict = dict(enumerate(labels))
                    parcellation_map[hem] = np.vectorize(transdict.get)(parcellation_map[hem])
    else:
        parcellation_map = {}
        for hem in ['L', 'R']:
            # load parcellation map
            if space == 'bigbrain':
                if parcellation_map == 'aal':
                    parcellation_map[hem] = nilearn.surface.load_surf_data(fetch_aal()[hem])
                else:
                    parcellation_map[hem] = nilearn.surface.load_surf_data(
                        os.path.join(
                            SRC_DIR, 
                            f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation.label.gii')
                        )
            elif space == 'fsaverage':
                parcellation_map[hem], _, _ = nibabel.freesurfer.io.read_annot(
                        os.path.join(
                            SRC_DIR, 
                            f'{hem.lower()}h_{parcellation_name}.annot')
                    )
            # label parcellation map if indicated
            if not load_indices:
                if parcellation_name == 'brodmann':
                    # the source (fsaverage) is a gifti file
                    orig_gifti = nibabel.load(
                        os.path.join(
                            SRC_DIR,
                            f'{hem.lower()}h_{parcellation_name}.label.gii'
                        )
                    )
                    transdict = orig_gifti.labeltable.get_labels_as_dict()
                elif parcellation_name == 'aal':
                    with open(os.path.join(SRC_DIR,'aal_labels.txt'), 'r') as f:
                        aal_labels_str = f.read().replace('\t', '  ')
                    transdict = {}
                    for line in aal_labels_str.split('\n')[1:]:
                        parc, label, _ = re.match("^([0-9]+)\W+(\w+.\w)\W+([^']*)", line).groups()
                        transdict[int(parc)] = label
                else:
                    # for others its an annot file
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
                if space == 'bigbrain':
                    # select only the vertices corresponding to the downsampled ico5 surface
                    # Warning: For finer parcellation schemes such as sjh this approach leads to
                    # a few parcels being removed
                    mat = scipy.io.loadmat(
                        os.path.join(
                            SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-pial_downsampled.mat'
                        )
                    )
                    bb_downsample_indices = mat['bb_downsample'][:, 0]-1 #1-indexing to 0-indexing
                    parcellation_map[hem] = parcellation_map[hem][bb_downsample_indices]
                elif space == 'fsaverage':
                    parcellation_map[hem] = parcellation_map[hem][:N_VERTICES_HEM_BB_ICO5]
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

    Returns
    -------
    labels: (np.ndarray)
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

def fetch_conn_matrix(kind, parcellation_name='schaefer400', dataset='hcp'):
    """
    Loads FC or SC matrices in Schaefer parcellation (400) from ENIGMA toolbox
    and EC matrix from Paquola 2021 (https://www.biorxiv.org/content/10.1101/2021.11.22.469533v1)
    and reorders them according to the order in other matrices (alphabetically,
    which is enforced in helpers.parcellate).

    Parameters
    ----------
    kind: (str)
        - structural
        - functional
        - effective
    parcellation_name: (str)
    dataset: (str) for the effective connectivity
        - hcp
        - mics

    Returns
    ---------
    reordered_conn_matrix, (np.ndarray) (n_parc, n_parc) 
        reordered FC or SC matrices matching the original matrix
    """
    if kind == 'effective':
        if (parcellation_name != 'schaefer400'):
            raise NotImplementedError
        # load the rDCM results from Paquola 2021
        results = scipy.io.loadmat(
            os.path.join(SRC_DIR, f'{dataset}_rDCM_sch400.mat'), 
            mat_dtype=True, struct_as_record=False)['results'][0][0]
        conn_matrix = results.mean_Amatrix_allSubjects
        # get the absolute values and zero out the diagonal
        # following Paquola 2021
        conn_matrix = np.abs(conn_matrix)
        conn_matrix[np.diag_indices_from(conn_matrix)] = 0
        conn_matrix_labels = [a[0] for a in results.AllRegions[0]]
    else:
        # match parcellation name with the enigmatoolbox
        ENIGMATOOLBOX_PARC_NAMES = {
            'aparc': 'aparc',
            'schaefer400': 'schaefer_400',
        }
        enigma_parcellation_name = ENIGMATOOLBOX_PARC_NAMES.get(parcellation_name, None)
        if enigma_parcellation_name == None:
            raise Exception(f"ENIGMA Toolbox does not have connectivity data for the parcellation {parcellation_name}")
        # load data from enigma toolbox
        if kind == 'structural':
            conn_matrix, conn_matrix_labels, _, _ = enigmatoolbox.datasets.load_sc(enigma_parcellation_name)
        else:
            conn_matrix, conn_matrix_labels, _, _ = enigmatoolbox.datasets.load_fc(enigma_parcellation_name)
            # in enigma toolbox negative values are zeroed out => setting them to nan
            # to ignore them in the analyses
            conn_matrix[conn_matrix==0] = np.nan
    conn_matrix = pd.DataFrame(conn_matrix, columns=conn_matrix_labels, index=conn_matrix_labels)
    # parcellate dummy data to get the order of parcels in other matrices created locally
    dummy_surf_data = np.zeros(N_VERTICES_HEM_BB * 2)
    parcellated_dummy = helpers.parcellate(dummy_surf_data, parcellation_name).dropna()
    # reorder matrices downloaded from enigmaltoolbox
    reordered_conn_matrix = conn_matrix.loc[parcellated_dummy.index, parcellated_dummy.index]
    return reordered_conn_matrix

def load_macaque_hierarchy():
    """
    Hierarchy of macaque parcels based on Burt 2018 in M132
    parcellation (https://www.nature.com/articles/s41593-018-0195-0)
    """
    # load and label the parcellated hierarchy map
    cifti = nibabel.load(os.path.join(
        SRC_DIR, 'macaque_hierarchy.pscalar.nii'
    ))
    hierarchy = pd.Series(
        cifti.get_fdata()[0, :], 
        index=cifti.header.get_axis(1).name
        )
    # rename some parcels to the names that are
    # in the M132 parcellation file
    rename_dict = {
        '29/30': '29_30',
        '9/46d': '9_46d',
        '9/46v': '9_46v',
        'CORE': 'Core',
        'ENTORHINAL': 'Ento',
        'INSULA': 'Ins',
        'OPRO': 'Opro',
        'PERIRHINAL': 'Peri',
        'PIRIFORM': 'Pir',
        'Parainsula': 'Pi',
        'Pro.St': 'Pro. St.',
        'SII': 'S2',
        'SUBICULUM': 'Sub',
        'TEMPORAL_POLE': 'TEMPORAL-POLE',
        'TEa/ma': 'TEa_m-a',
        'TEa/mp': 'TEa_m-p'
        }
    hierarchy = hierarchy.rename(index=rename_dict)
    # assign them to the left hemisphere
    hierarchy.index = 'L_' + hierarchy.index.to_series()
    return hierarchy

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
    **abagen_kwargs: see abagen.get_expression_data

    Returns
    -------
    ahba_data: (pd.DataFrame | dict) 
    """
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
    if parcellation_name == 'aparc':
        # with the original dk parcellation from freesurfer
        # abagen throws an error, so I will use the dk parcellation
        # that is included in abagen
        aparc = abagen.fetch_desikan_killiany(surface=True)
        atlas = aparc['image']
        atlas_info = pd.read_csv(aparc['info'], index_col='id')
    else:
        atlas = (os.path.join(SRC_DIR, f'lh_{parcellation_name}_fsa5.label.gii'),
                os.path.join(SRC_DIR, f'rh_{parcellation_name}_fsa5.label.gii'))
        _, atlas_info = abagen.images.check_surface(atlas)
    if return_donors:
        # avoid normalizing across samples before aggregate for subtypes are calculated
        gene_norm = None
    else:
        gene_norm = 'srs'
    # get the data as dict
    expression_data = abagen.get_expression_data(
        atlas = atlas,
        atlas_info = atlas_info,
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
    if parcellation_name == 'sjh':
        # remove sjh_ from the lable
        atlas_info['label'] = atlas_info['label'].map(lambda l: int(l.replace('sjh_','')))
    elif parcellation_name == 'aparc':
        # add hemisphere to the labels
        atlas_info['label'] = atlas_info['hemisphere'] + '_' + atlas_info['label']
    # exclude midline parcels
    parcels_mask = ~(atlas_info['label'].isin(helpers.MIDLINE_PARCELS[parcellation_name]))
    if parcellation_name == 'aparc':
        # aparc from abagen also includes subcortical parcels: remove them
        parcels_mask = parcels_mask & (atlas_info['structure']=='cortex')
    if discard_rh:
        # additionally exclude R hemisphere parcels
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
                                    ibf_threshold=0.5, **abagen_kwargs):
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
        limit the map to the left hemisphere. For consistency with other 
        functions the right hemisphere vertices/parcels are not removed 
        but are set to NaN
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
    **abagen_kwargs: see abagen.get_expression_data
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
    print(f'{gene_list.shape[0] - len(exist_gene_list)} of {gene_list.shape[0]} genes do not exist')
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

def fetch_laminar_cellular_features(parcellation_name=None):
    """
    Fetches the layer-specific cellular-level data from selected
    samples of the BigBrain, including cellular density, size
    and count.
    Data is available in 
    https://search.kg.ebrains.eu/instances/f06a2fd1-a9ca-42a3-b754-adaa025adb10

    Parameters
    ---------
    parcellation_name: (str | None)
        if provided all the samples within a parcel are merged

    Returns
    ------
    samples: (pd.DataFrame)
    density: (pd.DataFrame)
    size: (pd.DataFrame)
    count: (pd.DataFrame)
    """
    # load the csv files if they've been created before
    cellular_features_paths = {}
    for name in ['samples', 'density', 'size', 'count']:
        cellular_features_paths[name] = os.path.join(SRC_DIR, f'tpl-bigbrain_parc-{parcellation_name}_desc-cellular_{name}.csv')
    cellular_features = {}
    if all([os.path.exists(path) for path in cellular_features_paths.values()]):
        for name, path in cellular_features_paths.items():
            cellular_features[name] = pd.read_csv(path, index_col=0)
        return cellular_features.values()
    # importing siibra within the function as it's a big package
    import siibra
    # download the cellular features data
    atlas = siibra.atlases['human']
    julich_brain = atlas.get_parcellation('julich 2.9')
    all_features = siibra.get_features(julich_brain, siibra.modalities.CorticalCellDistribution)
    for name in cellular_features_paths.keys():
        cellular_features[name] = pd.DataFrame()
    # extract the data needed from each sample
    for i, feature in enumerate(all_features):
        cellular_features['samples'].loc[i, ['x', 'y', 'z']] = feature.location.coordinate
        cellular_features['samples'].loc[i, 'brain_area'] = feature.info['brain_area']
        cellular_features['samples'].loc[i, 'section_id'] = feature.info['section_id']
        for layer in range(1, 7):
            cellular_features['density'].loc[i, layer] = feature.layer_density(layer)
        cellular_features['size'].loc[i, range(1, 7)] = feature.cells.groupby('layer')['area'].mean().loc[1:7]
        cellular_features['count'].loc[i, range(1, 7)] = feature.cells.groupby('layer').count().loc[1:7, 'instance label']
    # assign each sample to its closest vertex on bigbrain surface
    for idx, row in cellular_features['samples'].iterrows():
        if row.iloc[0] >= 0:
            hemi = 'R'
        else:
            hemi = 'L'
        cellular_features['samples'].loc[idx, 'bb_vertex'] = np.linalg.norm(
            row.iloc[:3].values[np.newaxis, :].astype('float16')
            - nilearn.surface.load_surf_mesh(load_mesh_paths()[hemi]).coordinates, 
            keepdims=True, axis=1
        ).argmin()
        if hemi == 'R':
            cellular_features['samples'].loc[idx, 'bb_vertex'] += N_VERTICES_HEM_BB_ICO5
    cellular_features['samples']['bb_vertex'] = cellular_features['samples']['bb_vertex'].astype('int')
    # assign each vertex to a parcel if parcellation_name is provided
    if parcellation_name:
        cellular_features['samples']['parc'] = load_parcellation_map(parcellation_name, True, downsampled=True)[cellular_features['samples']['bb_vertex']]
        grouper = 'parc'
    else:
        grouper = 'bb_vertex'
    # merge multiple samples per vertex (31 of 111 samples)
    unmerged_samples = cellular_features['samples'].copy()
    cellular_features['samples'] = cellular_features['samples'].groupby(grouper).mean()
    for name in ['density', 'size', 'count']:
        cellular_features[name] = pd.concat(
            [cellular_features[name], unmerged_samples[grouper]]
            , axis=1).groupby(grouper).mean()
    # save
    for name, df in cellular_features.items():
        df.to_csv(cellular_features_paths[name])
    return cellular_features.values()

def fetch_aal():
    out_giis = {
        'L': os.path.join(SRC_DIR, 'tpl-bigbrain_hemi-L_desc-aal_parcellation.label.gii'),
        'R': os.path.join(SRC_DIR, 'tpl-bigbrain_hemi-R_desc-aal_parcellation.label.gii')
    }
    if os.path.exists(out_giis['L']) & os.path.exists(out_giis['R']):
        return out_giis
    aal_civet = {
        'L': np.loadtxt('https://github.com/aces/CIVET/raw/master/models/icbm/AAL/icbm_avg_mid_mc_AAL_left.txt'),
        'R': np.loadtxt('https://github.com/aces/CIVET/raw/master/models/icbm/AAL/icbm_avg_mid_mc_AAL_right.txt')
    }
    aal_civet_gii ={
        'L': neuromaps.images.construct_shape_gii(aal_civet['L']),
        'R': neuromaps.images.construct_shape_gii(aal_civet['R']),
    }
    aal_fsa_gii_paths = {
        'L': os.path.join(SRC_DIR, 'lh_aal.label.gii'),
        'R': os.path.join(SRC_DIR, 'rh_aal.label.gii')
    }
    for hemi in ['L', 'R']:
        aal_fsa_gii = neuromaps.transforms.civet_to_fsaverage(
            aal_civet_gii[hemi], target_density='164k', hemi=hemi, method='nearest'
            )[0]
        aal_fsa_gii.to_filename(aal_fsa_gii_paths[hemi])
    helpers.surface_to_surface_transform(
        SRC_DIR, SRC_DIR,
        'lh_aal.label.gii', 'rh_aal.label.gii',
        'fsaverage', 'bigbrain',
        'aal_parcellation')
    return out_giis