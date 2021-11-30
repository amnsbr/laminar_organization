import os
import numpy as np
import nilearn.surface
from cortex.polyutils import Surface
import subprocess
import nibabel
import scipy.spatial.distance
import pandas as pd
import helpers

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

def create_curvature_similarity_matrix(parcellation_name):
    """
    Creates parcel-to-parcel curvature similarity matrix based on the
    `parcellation_name` and by taking mean or median (`averaging_method`) of vertices

    parcellation_name: (str) name of the parcellation (must exist in `data/parcellation`)
    """
    out_path = os.path.join(DATA_DIR, 'matrix', f'curvature_similarity_parc-{parcellation_name}.txt')
    if os.path.exists(out_path):
        # skip this if GD matrix already exist
        print(f"Curvature similiarity matrix exists in {out_path}")
        return
    #> load curvature maps and determine min & max curvature across hems
    curvature = {}
    for hem in ['L', 'R']:
        curvature_filepath = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-{hem}_desc-mean_curvature.npy'
            )
        curvature[hem] = np.load(curvature_filepath)
    #> parcellate curvature
    parcellated_curvature = helpers.parcellate(
        curvature, 
        parcellation_name, 
        averaging_method=None
        )
    #> create pdfs and store min and max curv at each parcel
    pdfs = {}
    for hem in ['L', 'R']:
        pdfs[hem] = parcellated_curvature[hem].apply(
            lambda group_df: pd.Series({
                'pdf': scipy.stats.gaussian_kde(group_df[0]),
                'min_curv': group_df[0].min(),
                'max_curv': group_df[0].max(),
            })
            )
    #> concatenate L and R hemisphere parcels,
    #  dropping parcels duplicated in both hemispheres
    pdfs = (pd.concat([pdfs['L'], pdfs['R']], axis=0)
            .reset_index(drop=False)
            .drop_duplicates('index')
            .set_index('index')
            .reset_index(drop=True) # get rid of original parcel labels
    )
    #> measure parcel-to-parcel similarity of curvature distributions
    #  using Jensen-Shannon distance
    js_distance_matrix = np.zeros((pdfs.shape[0],pdfs.shape[0]))
    for idx_i, pdf_i in pdfs.iterrows():
        for idx_j, pdf_j in pdfs.iterrows():
            if idx_i == idx_j:
                js_distance_matrix[idx_i, idx_j] = 0
            elif idx_i > idx_j: # lower triangle only
                #> find the min and max curv across the pair of parcels and 
                #  create a linearly spaced discrete array [min, max]
                #  used for sampling PDFs of curvature in each parcel
                pair_min_curv = min(pdf_i['min_curv'], pdf_j['min_curv'])
                pair_max_curv = min(pdf_i['max_curv'], pdf_j['max_curv'])
                X_pair = np.linspace(pair_min_curv, pair_max_curv, 200)
                #> sample PDFi and PDFj at X_pair and convert it to discrete
                #  PDF via dividing by sum
                Y_i = pdf_i['pdf'].evaluate(X_pair)
                Y_j = pdf_j['pdf'].evaluate(X_pair)
                Y_i /= Y_i.sum()
                Y_j /= Y_j.sum()
                js_distance_matrix[idx_i, idx_j] = scipy.spatial.distance.jensenshannon(Y_i, Y_j)    #> calcualte curvature similarity as 1 - distance (TODO: is this the best approach?)
    #> make sure that there are no np.infs and the distance is bound by 0 and 1
    assert (js_distance_matrix.min() >= 0) and (js_distance_matrix.max() <= 1)
    #> calcualate similarity as 1 - dintance
    curv_similarity_matrix = 1 - js_distance_matrix
    #> copy the lower triangle to the upper triangle
    i_upper = np.triu_indices(curv_similarity_matrix.shape[0], 1)
    curv_similarity_matrix[i_upper] = curv_similarity_matrix.T[i_upper]
    np.savetxt(out_path, curv_similarity_matrix)
    print(f"Curvature similiarity matrix created in {out_path}")

def create_pairwise_geodesic_distance_matrix(parcellation_name, fill_l2r=True):
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
    outPath = os.path.join(DATA_DIR, 'matrix', f'geodesic_parc-{parcellation_name}_{"l2r_filled" if fill_l2r else ""}')
    if os.path.exists(outPath+'.txt'):
        # skip this if GD matrix already exist
        print(f"GD matrix exists in {outPath+'.txt'}")

    GDs = {}
    for hem in ['L', 'R']:
        #> load surf
        surf_path = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-L_desc-mid.surf.gii'
            )
        surf = nibabel.load(surf_path)
        vertices = surf.agg_data('NIFTI_INTENT_POINTSET')
        faces = surf.agg_data('NIFTI_INTENT_TRIANGLE')
        
        #> load parcellation map
        labels = nilearn.surface.load_surf_data(
            os.path.join(
                DATA_DIR, 'parcellation', 
                f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation.label.gii')
            )
        #> label parcellation map
        _, _, names = nibabel.freesurfer.io.read_annot(
            os.path.join(
                DATA_DIR, 'parcellation', 
                f'{hem.lower()}h.{parcellation_name}.annot')
        )
        if parcellation_name == 'sjh':
            names = list(map(lambda l: int(l.decode().replace('sjh_','')), names))
        transdict = dict(enumerate(names))
        parc = np.vectorize(transdict.get)(labels)
        # find centre vertices
        uparcel = np.unique(parc)
        voi = np.zeros([1, len(uparcel)])
        
        print(f"[ INFO ]..... Finings centre vertex for each parcel in hemisphere {hem}")
        for (i, parcel_name) in enumerate(uparcel):
            this_parc = np.where(parc == parcel_name)[0]
            distances = scipy.spatial.distance.pdist(np.squeeze(vertices[this_parc,:]), 'euclidean') # Returns condensed matrix of distances
            distancesSq = scipy.spatial.distance.squareform(distances) # convert to square form
            sumDist = np.sum(distancesSq, axis = 1) # sum distance across columns
            index = np.where(sumDist == np.min(sumDist)) # minimum sum distance index
            voi[0, i] = this_parc[index[0][0]]
            
        # Initialize distance matrix
        GDs[hem] = np.zeros((uparcel.shape[0], uparcel.shape[0]))

        print(f"[ INFO ]..... Running geodesic distance in hemisphere {hem}")
        for i in range(len(uparcel)):
            print("Parcel: ", uparcel[i])
            center_vertex = int(voi[0,i])
            cmdStr = f"{wbPath} -surface-geodesic-distance {surf_path} {center_vertex} {outPath}_this_voi.func.gii"
            subprocess.run(cmdStr.split())
            tmpname = outPath + '_this_voi.func.gii'
            tmp = nibabel.load(tmpname).agg_data()
            os.remove(tmpname)
            parcGD = np.empty((1, len(uparcel)))
            for n in range(len(uparcel)):
                tmpData = tmp[parc == uparcel[n]]
                tmpMean = np.mean(tmpData)
                parcGD[0, n] = tmpMean
            GDs[hem][i,:] = parcGD
        GDs[hem] = pd.DataFrame(GDs[hem], index=uparcel, columns=uparcel)
    #> join the GD matrices from left and right hemispheres
    GD_full = (pd.concat([GDs['L'], GDs['R']],axis=0)
            .reset_index(drop=False)
            .drop_duplicates('index')
            .set_index('index')
            .fillna(0))
    #> for SJH parcellation (and other parcellations with a midline overlapping parcel)
    #  calculate L to R GD as GD(L parcel, parcel 0)+GD(parcel 0, R parcel)
    if parcellation_name == 'sjh':
        #> calculate GD(L parcel, parcel 0)+GD(parcel 0, R parcel) for
        #  all pairs of (L parcel, R parcel)
        GD_l2r = GDs['L'][0].values[1:].reshape(-1, 1) \
                + GDs['R'][0].values[1:].reshape(1, -1)
        #> use this to fill in the matrix
        GD_full.iloc[1: GDs['L'].shape[0], GDs['L'].shape[0]:] = GD_l2r
        GD_full.iloc[GDs['L'].shape[0]:, 1:GDs['L'].shape[0]] = GD_l2r.T
        #> also fill in the (parcel 0, R parcel) GD which was removed
        #  while creating GD_full
        GD_full.iloc[0, GDs['L'].shape[0]:] = GDs['R'][0].iloc[1:]
        GD_full.iloc[GDs['L'].shape[0]:, 0] = GDs['R'][0].iloc[1:]
    np.savetxt(outPath+'.txt', GD_full, fmt='%.12f')
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
        print(f"Creating the curvature similarity matrix for {parcellation_name} parcellation")
        create_curvature_similarity_matrix(parcellation_name)
        print(f"Creating pairwise geodesic distance matrix for {parcellation_name} parcellation")
        create_pairwise_geodesic_distance_matrix(parcellation_name)

if __name__=='__main__':
    calculate_covariates()
