import os
import numpy as np
import nilearn.surface
from cortex.polyutils import Surface
import subprocess
import nibabel.load, nibabel.freesurfer.io
import scipy.spatial.distance

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')
MICAPIPE_BASE = os.path.join(cwd, '..', 'tools', 'micapipe')
wbPath = os.path.join(cwd, '..', 'tools', 'workbench', 'bin_linux64', 'wb_command')

def create_curvature_surf_map():
    """
    Creates the map of curvature for the bigbrain surface
    using pycortex
    """
    for hem in ['L', 'R']:
        curvature_filepath = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-{hem}_desc-mean_curvature.npy'
            )
        if os.path.exists(curvature_filepath):
            print(f"Curvature already exists in {curvature_filepath}")
            continue
        vertices, faces = nilearn.surface.load_surf_mesh(
            os.path.join(
                DATA_DIR, 'surface',
                f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                )
        )
        surface = Surface(vertices, faces)
        curvature = surface.mean_curvature()
        np.save(curvature_filepath, curvature)
        print(f"Curvature save in {curvature_filepath}")

def label2annot(label_path, outfile=None):
	"""
	Converts .label.gii to .annot

	Parameters
	----------
	label_path: (str) path to the .label.gii file
	outfile: (str) path to the output .annot file [Default: same as label_path with .annot ext]
	"""
	#> read labels (the map of parcel ids at each vertex)
	labels = nibabel.load(label_path).darrays[0].data
	n_labels = np.unique(labels).shape[0]
	#> create a pseudo color table (n_labels, 5) => [r, g, b, t, parcel_id] 
	pseudo_ctab = np.zeros((n_labels, 5))
	pseudo_ctab[:, :3] = np.vstack([np.zeros(n_labels), np.unique(labels) // 255, np.unique(labels)  % 255]).T
	pseudo_ctab[:, 4] = np.unique(labels)
	#> create pseudo names for parcels
	pseudo_names = [str(i) for i in sorted(list(np.unique(labels)))]
	#> write the .annot file
	if not outfile:
		outfile = label_path.replace('.label.gii', '.annot')
	nibabel.freesurfer.io.write_annot(outfile, labels, pseudo_ctab, pseudo_names)

def create_pairwise_geodesic_distance_matrix(parcellation_name):
    """
    Creates parcel-to-parcel geodesic distance matrix based on the
    parcellation ("parcellation_name")

    Parameters
    ----------
    parcellation_name: (str) name of the parcellation (must be stored in data/parcellations)

    Based on "geoDistMapper.py" from micapipe/functions (modified slightly)
    Credit:
    # Translated from matlab:
    # Original script by Boris Bernhardt and modified by Casey Paquola
    # Translated to python by Jessica Royer
    """
    # Set up
    outPath = os.path.join(DATA_DIR, 'matrix', f'geodesic_{parcellation_name}_parcellation')
    if os.path.exists(outPath):
        # skip this if GD matrix already exist
        print(f"GD matrix exists in {outPath}")
        return
    lh_surf = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-L_desc-mid.surf.gii'
        )
    rh_surf = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-R_desc-mid.surf.gii'
        )
    lh_annot = os.path.join(
            DATA_DIR, 'parcellation', 
            f'tpl-bigbrain_hemi-L_desc-{parcellation_name}_parcellation.annot'
        )
    rh_annot = os.path.join(
            DATA_DIR, 'parcellation', 
            f'tpl-bigbrain_hemi-L_desc-{parcellation_name}_parcellation.annot'
        )

    # Convert .label.gii to .annot if necessary
    if not os.path.exists(lh_annot):
        label2annot(lh_annot.replace('.annot', '.label.gii'))
        label2annot(rh_annot.replace('.annot', '.label.gii'))

    # Load surface
    lh = nibabel.load(lh_surf)
    vertices_lh = lh.agg_data('NIFTI_INTENT_POINTSET')
    faces_lh = lh.agg_data('NIFTI_INTENT_TRIANGLE')

    rh = nibabel.load(rh_surf)
    vertices_rh = rh.agg_data('NIFTI_INTENT_POINTSET')
    faces_rh = rh.agg_data('NIFTI_INTENT_TRIANGLE') + len(vertices_lh)

    vertices = np.append(vertices_lh, vertices_rh, axis = 0)
    faces = np.append(faces_lh, faces_rh, axis = 0)


    # Read annotation & join hemispheres
    [labels_lh, ctab_lh, names_lh] = nibabel.freesurfer.io.read_annot(lh_annot, orig_ids=True)
    [labels_rh, ctab_rh, names_rh] = nibabel.freesurfer.io.read_annot(rh_annot, orig_ids=True)
    nativeLength = len(labels_lh)+len(labels_rh)
    parc = np.zeros((nativeLength))
    for (x, _) in enumerate(labels_lh):
        parc[x] = np.where(ctab_lh[:,4] == labels_lh[x])[0][0]
    for (x, _) in enumerate(labels_rh):
        parc[x + len(labels_lh)] = np.where(ctab_rh[:,4] == labels_rh[x])[0][0] + len(ctab_lh)


    # Find centre vertex
    uparcel = np.unique(parc)
    voi = np.zeros([1, len(uparcel)])

    print("[ INFO ]..... Finings centre vertex for each parcel")
    for (n, _) in enumerate(uparcel):
        this_parc = np.where(parc == uparcel[n])[0]
        distances = scipy.spatial.distance.pdist(np.squeeze(vertices[this_parc,:]), 'euclidean') # Returns condensed matrix of distances
        distancesSq = scipy.spatial.distance.squareform(distances) # convert to square form
        sumDist = np.sum(distancesSq, axis = 1) # sum distance across columns
        index = np.where(sumDist == np.min(sumDist)) # minimum sum distance index
        voi[0, n] = this_parc[index[0][0]]


    # Initialize distance matrix
    GD = np.zeros((uparcel.shape[0], uparcel.shape[0]))


    # Left hemisphere
    parcL = parc[0:len(labels_lh)]
    print("[ INFO ]..... Running geodesic distance in the left hemisphere")
    for ii in range(len(np.unique(labels_lh))):
        vertex = int(voi[0,ii])
        voiStr = str(vertex)

        cmdStr = "{wbPath} -surface-geodesic-distance {lh_surf} {voiStr} {outPath}_this_voi.func.gii".format(wbPath=wbPath, lh_surf=lh_surf, voiStr=voiStr, outPath=outPath)
        subprocess.run(cmdStr.split())

        tmpname = outPath + '_this_voi.func.gii'
        tmp = nibabel.load(tmpname).agg_data()
        os.remove(tmpname)
        parcGD = np.empty((1, len(np.unique(labels_lh))))
        for n in range(len(np.unique(labels_lh))):
            tmpData = tmp[parcL == uparcel[n]]
            tmpMean = np.mean(tmpData)
            parcGD[0, n] = tmpMean
        GD[ii,:len(np.unique(labels_lh))] = parcGD


    # Right hemisphere
    parcR = parc[-len(labels_rh):]
    print("[ INFO ]..... Running geodesic distance in the right hemisphere")
    for ii in range(len(np.unique(labels_rh))):
        ii_rh = int(ii + len(uparcel)/2)
        vertex = int(voi[0,ii_rh] - len(vertices_lh))
        voiStr = str(vertex)

        cmdStr = "{wbPath} -surface-geodesic-distance {rh_surf} {voiStr} {outPath}_this_voi.func.gii".format(wbPath=wbPath, rh_surf=rh_surf, voiStr=voiStr, outPath=outPath)
        subprocess.run(cmdStr.split())

        tmpname = outPath + '_this_voi.func.gii'
        tmp = nibabel.load(tmpname).agg_data()
        os.remove(tmpname)
        parcGD = np.empty((1, len(np.unique(labels_rh))))
        for n in range(len(np.unique(labels_rh))):
            n_rh = int(n + len(uparcel)/2)
            tmpData = tmp[parcR == uparcel[n_rh]]
            tmpMean = np.mean(tmpData)
            parcGD[0, n] = tmpMean

        GD[ii_rh,-len(np.unique(labels_rh)):] = parcGD

    np.savetxt(outPath + '.txt', GD, fmt='%.12f')
    print("[ INFO ]..... Geodesic distance completed")        

def calculate_covariates():
    """
    Wrapper for functions that are used for creating the covariates
    """
    # 1) create the curvature map
    print("Creating the curvature map")
    create_curvature_surf_map()
    # 2) create pairwise geodesic distance matrices for each parcellation
    for parcellation_name in ['sjh']:
        print(f"Creating pairwise geodesic distance matrix for {parcellation_name} parcellation")
        create_pairwise_geodesic_distance_matrix(parcellation_name)

if __name__=='__main__':
    calculate_covariates()
