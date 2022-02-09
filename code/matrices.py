import os
import numpy as np
import nilearn.surface
import sklearn as sk
import scipy.stats
from cortex.polyutils import Surface
import subprocess
import nibabel
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm # color maps

import helpers
import datasets

#> specify directories
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
MICAPIPE_BASE = os.path.join(cwd, '..', 'tools', 'micapipe')
wbPath = os.path.join(cwd, '..', 'tools', 'workbench', 'bin_linux64', 'wb_command')

class MicrostructuralCovarianceMatrix:
    """
    Matrix showing microstructural similarity of parcels in their relative laminar
    thickness, relative laminar volume, density profiles (MPC), or their combination
    """
    def __init__(self, input_type, correct_curvature='volume',
                 similarity_metric = 'parcor', similarity_scale='parcel',
                 exc_masks=None,  parcellation_name='sjh'):
        """
        Initializes laminar similarity matrix object

        Parameters
        ---------
        input_type: (str) Type of input data
            - 'thickness' [default]: laminar thickness
            - 'density': profile density
            - 'thickness-density': fused laminar thickness and profile density
        correct_curvature: (str or None) ignored for 'density'
            - 'volume' [default]: normalize relative thickness by curvature according to the 
            equivolumetric principle i.e., use relative laminar volume instead of relative laminar 
            thickness. Laminar volume is expected to be less affected by curvature.
            - 'regress': regresses map of curvature out from the map of relative thickness
            of each layer (separately) using a linear regression. This is a more lenient approach of 
            controlling for curvature.
            - None
        similarity_metric: (str) how is similarity of laminar structure between two parcels determined
            - 'parcor': partial correlation with mean thickness pattern as covariate
            - 'euclidean': euclidean distance inverted and normalize to 0-1
            - 'pearson': Pearson's correlation coefficient
        similarity_scale: (str) granularity of similarity measurement
            - 'parcel' [default]: similarity method is used between average laminar profile of parcels
            - 'vertex': similarity method is used between all pairs of vertices between two
                        parcels and then the similarity metric is averaged
        exc_masks: (dict of np.ndarray) The surface masks of vertices that should be excluded (L and R)
        parcellation_name: (str) Parcellation scheme
            - 'sjh'
        """
        #> save parameters as class fields
        self.input_type = input_type
        self.correct_curvature = correct_curvature
        self.similarity_metric = similarity_metric
        self.similarity_scale = similarity_scale
        self.exc_masks = exc_masks
        self.parcellation_name = parcellation_name
        #> directory and filename (prefix which will be used for .npz and .jpg files)
        self.dir_path = self._get_dir_path()
        os.makedirs(self.dir_path, exist_ok=True)
        if os.path.exists(os.path.join(self.dir_path, 'matrix.csv')):
            self._load()
        else:
            self._create()
            self._save()
            self.plot()

    def _create(self):
        """
        Creates microstructural covariance matrix
        """
        print(f"""
        Creating microstructural covariance matrix:
            - input_type: {self.input_type},
            - correct_curvature: {self.correct_curvature}
            - parcellation_name: {self.parcellation_name},
            - exc_mask: {True if self.exc_masks else False}
        """)
        if self.input_type == 'thickness-density':
            matrices = []
            for input_type in ['thickness', 'density']:
                #> create matrix_obj for each input_type
                #  by calling the same class
                matrix_obj = MicrostructuralCovarianceMatrix(
                    input_type = input_type,
                    correct_curvature=self.correct_curvature,
                    similarity_metric=self.similarity_metric,
                    similarity_scale=self.similarity_scale,
                    exc_masks=self.exc_masks,
                    parcellation_name=self.parcellation_name
                )
                matrices.append(matrix_obj.matrix)
                #TODO: it is unlikely but maybe valid parcels are different for each modality
                # in that case this code won't work properly
                self.valid_parcels = matrix_obj.valid_parcels
            self.matrix = self._fuse_matrices(matrices)
        else:
            #> Load laminar thickness or density profiles
            print("Reading input")
            self._input_data = self._load_input_data()
            #> create the similarity matrix
            if self.similarity_scale == 'parcel':
                self.matrix, self.valid_parcels = self._create_at_parcels()
            else:
                self.matrix, self.valid_parcels = self._create_at_vertices()
            #> convert it to pd.DataFrame
            self.matrix = pd.DataFrame(
                self.matrix, 
                index=self.valid_parcels,
                columns=self.valid_parcels)

    def _load_input_data(self):
        """
        Load input data
        """
        if self.input_type == 'thickness':
            if self.correct_curvature == 'volume':
                input_data = datasets.load_laminar_volume(
                    exc_masks=self.exc_masks,
                )
            elif self.correct_curvature == 'regress':
                input_data = datasets.load_laminar_thickness(
                    exc_masks=self.exc_masks,
                    regress_out_curvature=True,
                    normalize_by_total_thickness=True,
                )
            else:
                input_data = datasets.load_laminar_thickness(
                    exc_masks=self.exc_masks,
                    regress_out_curvature=False,
                    normalize_by_total_thickness=True,
                )
        elif self.input_type == 'density':
            input_data = datasets.load_total_depth_density(
                exc_masks=self.exc_masks
            )
        return input_data


    def _create_at_parcels(self):
        """
        Creates laminar similarity matrix by taking Euclidean distance, Pearson's correlation
        or partial correltation (with the average laminar data pattern as the covariate) between
        average values of parcels

        Note: Partial correlation is based on "Using recursive formula" subsection in the wikipedia
        entry for "Partial correlation", which is also the same as Formula 2 in Paquola et al. PBio 2019
        (https://doi.org/10.1371/journal.pbio.3000284)
        Note 2: Euclidean distance is reversed (* -1) and rescaled to 0-1 (with 1 showing max similarity)

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                microstructure (laminar thickness or density profiles)
        """
        print("Concatenating and parcellating the data")
        self._parcellated_input_data = helpers.parcellate(
            self._input_data,
            self.parcellation_name,
            averaging_method='median'
            )
        #> concatenate left and right hemispheres
        self._concat_parcellated_input_data = helpers.concat_hemispheres(self._parcellated_input_data, dropna=True)
        print(f"Creating similarity matrix by {self.similarity_metric} at parcel scale")
        #> Calculate parcel-wise similarity matrix
        if self.similarity_metric in ['parcor', 'pearson']:
            if self.similarity_metric == 'parcor':
                #> calculate partial correlation
                r_ij = np.corrcoef(self._concat_parcellated_input_data)
                mean_input_data = self._concat_parcellated_input_data.mean()
                r_ic = self._concat_parcellated_input_data\
                            .corrwith(mean_input_data, 
                            axis=1) # r_ic and r_jc are the same
                r_icjc = np.outer(r_ic, r_ic) # the second r_ic is actually r_jc
                matrix = (r_ij - r_icjc) / np.sqrt(np.outer((1-r_ic**2),(1-r_ic**2)))
            else:
                np.corrcoef(self._concat_parcellated_input_data.values)
            #> zero out negative correlations
            matrix[matrix<0] = 0
            #> zero out correlations of 1 (to avoid division by 0)
            matrix[matrix==1] = 0
            #> Fisher's z-transformation
            matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
            #> zero out NaNs and inf
            matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        elif self.similarity_metric == 'euclidean':
            #> calculate pair-wise euclidean distance
            matrix = sk.metrics.pairwise.euclidean_distances(self._concat_parcellated_input_data.values)
            #> make it negative (so higher = more similar) and rescale to range (0, 1)
            matrix = sk.preprocessing.minmax_scale(-matrix, (0, 1))
        #> determine valid parcels
        valid_parcels = self._concat_parcellated_input_data.index.tolist()
        return matrix, valid_parcels

    def _create_at_vertices(self):
        """
        Creates laminar similarity matrix by taking Euclidean distance or Pearson's correlation
        between all pairs of vertices between two pairs of parcels and then taking the average value of
        the resulting n_vert_parcel_i x n_vert_parcel_j matrix for the cell i, j of the parcels similarity matrix

        Note: Average euclidean distance is reversed (* -1) and rescaled to 0-1 (with 1 showing max similarity)

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                laminar data pattern
        """
        if self.similarity_metric != 'euclidean':
            raise NotImplementedError("Correlation at vertex level is not implemented")
        #> Concatenate and parcellate the data
        print("Concatenating and parcellating the data")
        concat_input_data = np.concatenate([self._input_data['L'], self._input_data['R']], axis=0)
        self._concat_parcellated_input_data = helpers.parcellate(
            concat_input_data,
            self.parcellation_name,
            averaging_method=None
            ) # a groupby object
        n_parcels = len(self._concat_parcellated_input_data)
        #> Calculating similarity matrix
        print(f"Creating similarity matrix by {self.similarity_metric} at vertex scale")
        matrix = pd.DataFrame(
            np.zeros((n_parcels, n_parcels)),
            columns = self._concat_parcellated_input_data.groups.keys(),
            index = self._concat_parcellated_input_data.groups.keys()
        )
        invalid_parcs = []
        for parc_i, vertices_i in self._concat_parcellated_input_data.groups.items():
            print("\tParcel", parc_i) # printing parcel_i name since it'll take a long time per parcel
            input_data_i = concat_input_data[vertices_i,:]
            input_data_i = input_data_i[~np.isnan(input_data_i).any(axis=1)]
            if input_data_i.shape[0] == 0: # sometimes all values may be NaN
                matrix.loc[parc_i, :] = np.NaN
                invalid_parcs.append(parc_i)
                continue
            for parc_j, vertices_j in self._concat_parcellated_input_data.groups.items():
                input_data_j = concat_input_data[vertices_j,:]
                input_data_j = input_data_j[~np.isnan(input_data_j).any(axis=1)]
                if input_data_i.shape[0] == 0:
                    matrix.loc[parc_i, :] = np.NaN
                else:
                    matrix.loc[parc_i, parc_j] = sk.metrics.pairwise.euclidean_distances(input_data_i, input_data_j).mean()
        #> make ED values negative (so higher = more similar) and rescale to range (0, 1)
        matrix = sk.preprocessing.minmax_scale(-matrix, (0, 1))
        #> store the valid parcels
        valid_parcels = sorted(list(set(self._concat_parcellated_input_data.groups.keys())-set(invalid_parcs)))
        return matrix, valid_parcels

    def _fuse_matrices(self, matrices):
        """
        Fuses two input matrices by rank normalizing each and rescaling
        the second matrix based on the first one (which shouldn't make much
        difference if the matrices are rank normalized; TODO: think about this)

        Parameters
        ----------
        matrices: (list of pd.DataFrame) matrices with the same shape. The code
            can handle more than two matrices but that doesn't occur normally!

        Note: This function doesn't need to be in this class but is included here
        anyway because it's not used anywhere else!

        Based on Paquola 2020 PBio
        https://github.com/MICA-MNI/micaopen/tree/master/structural_manifold
        Note: This function does not zero out negative or NaNs and assumes
        all processes are completed on the input matrices
        """
        rank_normalized_matrices = []
        #> rank normalize the first matrix
        rank_normalized_matrices.append(
            pd.DataFrame(
                scipy.stats.rankdata(matrices[0].values)
                .reshape(matrices[0].shape)
            )
        )
        #> rank normalize and rescale next matrices
        for matrix in matrices[1:]:
            #> rank normalize (and flatten) the matrix
            rank_normalized_matrix_flat = scipy.stats.rankdata(matrix.values)
            #> rescale it by the first matrix
            rank_normalized_rescaled_matrix_flat = np.interp(
                rank_normalized_matrix_flat,
                (rank_normalized_matrix_flat.min(), rank_normalized_matrix_flat.max()),
                (rank_normalized_matrices[0].values.min(), rank_normalized_matrices[0].values.max())
            )
            rank_normalized_matrices.append(
                pd.DataFrame(
                    rank_normalized_rescaled_matrix_flat.reshape(matrix.shape)
                )
            )
        #> fuse the matrices horizontally as a pd.DataFrame and label it
        fused_matrix = pd.concat(rank_normalized_matrices, axis=1)
        fused_matrix.index = matrices[0].index
        fused_matrix.columns = matrices[0].index.tolist() * len(matrices)
        return fused_matrix

    def _get_dir_path(self):
        """
        Get path for the matrix directory
        """
        MAIN_DIRS = {
            'thickness-density': 'fused',
            'thickness': 'ltc',
            'density': 'mpc'
        }
        main_dir = MAIN_DIRS[self.input_type]
        sub_dir = f'parc-{self.parcellation_name}' \
            + f'_curv-{str(self.correct_curvature).lower()}'\
            + ('_exc-adys' if self.exc_masks else '')\
            + f'_metric-{self.similarity_metric}'\
            + f'_scale-{self.similarity_scale}'
        return os.path.join(OUTPUT_DIR, main_dir, sub_dir)
        
    def _save(self):
        """
        Save the matrix to a .csv file
        """
        self.matrix.to_csv(
            os.path.join(self.dir_path, 'matrix.csv'), 
            index_label='parcel'
        )

    def _load(self):
        """
        Loads a matrix created before
        """
        self.matrix = pd.read_csv(
            os.path.join(self.dir_path, 'matrix.csv'),
            index_col='parcel'
        )
        self.valid_parcels = self.matrix.index.tolist()
        # for sjh parcellation index is read as 'int' but columns
        # are read as str. Fix this by making columns similar to
        # index * number of matrices
        n_sq_matrix = self.matrix.shape[1] // self.matrix.shape[0]
        self.matrix.columns = self.valid_parcels * n_sq_matrix

    def plot(self):
        """
        Plot the matrix as heatmap
        """
        #> use different colors for different input types
        if self.input_type == 'thickness':
            cmap = sns.color_palette("RdBu_r", as_cmap=True)
        elif self.input_type == 'density':
            cmap = sns.color_palette("rocket", as_cmap=True)
        else:
            # TODO: use different colors for each input type in the plot
            cmap = sns.color_palette("RdBu_r", as_cmap=True)
        #> plot the matrix
        helpers.plot_matrix(
            self.matrix.values,
            os.path.join(self.dir_path,'matrix'),
            cmap=cmap,
            vrange=(0.025, 0.975)
            )

class CurvatureSimilarityMatrix:
    """
    Matrix showing similarity of curvature distribution
    between each pair of parcels
    """
    def __init__(self, parcellation_name):
        """
        Creates curvature similarity matrix or loads it if it already exist

        Parameters
        ----------
        parcellation_name: (str)
        """
        self.parcellation_name = parcellation_name
        self.path = os.path.join(OUTPUT_DIR, 'curvature', f'curvature_similarity_matrix_parc-{parcellation_name}.csv')
        if os.path.exists(self.path):
            self.matrix = pd.read_csv(
                self.path,
                index_col='parcel'
            )
        else:
            os.makedirs(os.path.join(OUTPUT_DIR, 'curvature'), exist_ok=True)
            self.matrix = self._create()
        
    def _create(self):
        """
        Create curvature similarity matrix by calculating Jansen-Shannon similarity of
        curvature distributions between pairs of parcels        
        """
        #> load curvature maps
        curvature_maps = datasets.load_curvature_maps()
        #> parcellate curvature
        parcellated_curvature = helpers.parcellate(
            curvature_maps, 
            self.parcellation_name, 
            averaging_method=None,
            na_midline=False,
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
        )
        #> measure parcel-to-parcel similarity of curvature distributions
        #  using Jensen-Shannon distance
        js_distance_matrix = pd.DataFrame(
            np.zeros((pdfs.shape[0],pdfs.shape[0])),
            columns=pdfs.index,
            index=pdfs.index
        )
        for parc_i, pdf_i in pdfs.iterrows():
            for parc_j, pdf_j in pdfs.iterrows():
                if parc_i == parc_j:
                    js_distance_matrix.loc[parc_i, parc_j] = 0
                elif parc_i > parc_j: # lower triangle only
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
                    js_distance_matrix.loc[parc_i, parc_j] = scipy.spatial.distance.jensenshannon(Y_i, Y_j)    #> calcualte curvature similarity as 1 - distance (TODO: is this the best approach?)
        #> make sure that there are no np.infs and the distance is bound by 0 and 1
        assert (js_distance_matrix.values.min() >= 0) and (js_distance_matrix.values.max() <= 1)
        #> calcualate similarity as 1 - dintance
        curv_similarity_matrix = 1 - js_distance_matrix
        #> copy the lower triangle to the upper triangle
        i_upper = np.triu_indices(curv_similarity_matrix.shape[0], 1)
        curv_similarity_matrix.values[i_upper] = curv_similarity_matrix.T.values[i_upper]
        #> save it
        curv_similarity_matrix.to_csv(self.path, index_label='parcel')
        return curv_similarity_matrix

class GeodesicDistanceMatrix:
    """
    Matrix of geodesic distance between centroids of parcels
    """
    def __init__(self, parcellation_name, approach='center-to-center'):
        """
        Creates parcel-to-parcel geodesic distance matrix based on the
        parcellation ("parcellation_name") or loads it if it already exists

        Parameters
        ----------
        parcellation_name: (str) name of the parcellation (must be stored in data/parcellations)
        approach: (str)
            - center-to-center: calculate pair-wise distance between centroids of parcels. Results in symmetric matrix.
            - center-to-parcel: calculates distance between centroid of one parcel to all vertices
                                in the other parcel, taking the mean distance. Can result in asymmetric matrix.
                                (this is "geoDistMapper.py" behavior)

        Based on "geoDistMapper.py" from micapipe/functions
        Original Credit:
        # Translated from matlab:
        # Original script by Boris Bernhardt and modified by Casey Paquola
        # Translated to python by Jessica Royer
        """
        self.parcellation_name = parcellation_name
        self.approach = approach
        self.path = os.path.join(
            OUTPUT_DIR, 'geodesic_distance',
            f'geodesic_distance_matrix_parc-{parcellation_name}_approach-{approach}.csv')
        if os.path.exists(self.path):
            self.matrix = pd.read_csv(
                self.path,
                index_col='parcel'
            )
        else:
            os.makedirs(os.path.join(OUTPUT_DIR, 'geodesic_distance'), exist_ok=True)
            self.matrix = self._create()

    def _create(self):
        """
        Creates center-to-parcel or center-to-center geodesic distance matrix
        between pairs  of parcels
        """
        GDs = {}
        for hem in ['L', 'R']:
            #> load surf
            surf_path = os.path.join(
                SRC_DIR, 
                f'tpl-bigbrain_hemi-L_desc-mid.surf.gii'
                )
            surf = nibabel.load(surf_path)
            vertices = surf.agg_data('NIFTI_INTENT_POINTSET')
            
            #> load parcellation map
            labels = nilearn.surface.load_surf_data(
                os.path.join(
                    SRC_DIR, 
                    f'tpl-bigbrain_hemi-{hem}_desc-{self.parcellation_name}_parcellation.label.gii')
                )
            #> label parcellation map
            _, _, names = nibabel.freesurfer.io.read_annot(
                os.path.join(
                    SRC_DIR, 
                    f'{hem.lower()}h_{self.parcellation_name}.annot')
            )
            if self.parcellation_name == 'sjh':
                names = list(map(lambda l: int(l.decode().replace('sjh_','')), names))
            elif self.parcellation_name == 'aparc':
                names = list(map(lambda l: f'{hem}_{l.decode()}', names)) # so that it matches ENIGMA toolbox dataset
            elif self.parcellation_name in ['schaefer400', 'economo']:
                names = list(map(lambda l: l.decode(), names)) # b'name' => 'name'
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
                cmdStr = f"{wbPath} -surface-geodesic-distance {surf_path} {center_vertex} {self.path}_this_voi.func.gii"
                subprocess.run(cmdStr.split())
                tmpname = self.path + '_this_voi.func.gii'
                tmp = nibabel.load(tmpname).agg_data()
                os.remove(tmpname)
                parcGD = np.empty((1, len(uparcel)))
                for n in range(len(uparcel)):
                    if self.approach=='center-to-parcel':
                        tmpData = tmp[parc == uparcel[n]]
                        pairGD = np.mean(tmpData)
                    elif self.approach=='center-to-center':
                        other_center_vertex = int(voi[0, n])
                        pairGD = tmp[other_center_vertex]
                    parcGD[0, n] = pairGD
                GDs[hem][i,:] = parcGD
            #> save the GD for the current hemisphere
            np.savetxt(
                self.path.replace('_parc', f'_hemi-{hem}_parc'),
                GDs[hem],
                fmt='%.12f')
            #> convert it to dataframe so that joining hemispheres would be easier
            GDs[hem] = pd.DataFrame(GDs[hem], index=uparcel, columns=uparcel)
        #> join the GD matrices from left and right hemispheres
        GD_full = (pd.concat([GDs['L'], GDs['R']],axis=0)
                .reset_index(drop=False)
                .drop_duplicates('index')
                .set_index('index')
                .fillna(0))
        GD_full.to_csv(self.path, index_label='parcel')
        return GD_full

