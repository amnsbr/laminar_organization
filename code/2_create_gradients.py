import os
import pandas as pd
import numpy as np
import sklearn as sk
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.surface
import brainspace.gradient
import helpers

#> specify the data dir and create gradients and matrices subfolders
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')
os.makedirs(os.path.join(DATA_DIR, 'gradient'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'matrix'), exist_ok=True)
#> specify the path to adysgranular masks
adysgranular_masks = {
    'L': os.path.join(
        DATA_DIR, 'surface',
        'tpl-bigbrain_hemi-L_desc-adysgranular_mask_parcellation-sjh_thresh_0.1.npy'
    ),
    'R': os.path.join(
        DATA_DIR, 'surface',
        'tpl-bigbrain_hemi-R_desc-adysgranular_mask_parcellation-sjh_thresh_0.1.npy'
    )
}


#> laminar similarity matrix class
class LaminarSimilarityMatrix:
    def __init__(self, input_type, out_dir, normalize_by_total_thickness=True,
                 similarity_method = 'partial_corr', similarity_scale='parcel',
                 exc_masks=None,  parcellation_name='sjh', correct_thickness_by_curvature=True):
        """
        Initializes laminar thickness/density similarity matrix object

        Parameters
        ---------
        input_type: (str) Type of input laminar data
            - 'thickness' (default)
            - 'density'
        out_dir: (str) path to the output directory
        normalize_by_total_thickness: (bool) Normalize by total thickness. Default: True
        similarity_method: (str) how is similarity of laminar structure between two parcels determined
            - 'partial_corr': partial correlation with mean thickness pattern as covariate
            - 'euclidean': euclidean distance [will normalize to 0-1]; very similar to pearson
            - 'pearson': Pearson's correlation coefficient
        similarity_scale: (str) granularity of similarity measurement
            - 'parcel': similarity method is used between average laminar profile of parcels
            - 'vertex': similarity method is used between all pairs of vertices between two
                        parcels and then the similarity metric is averaged
        exc_masks: (dict of str) Path to the surface masks of vertices that should be excluded (L and R) (format: .npy)
        correct_thickness_by_curvature: (bool) Regress out curvature. Default: True
        parcellation_name: (str) Parcellation scheme
            - 'sjh'
        """
        #> save parameters as class fields
        self.input_type = input_type
        self.normalize_by_total_thickness = normalize_by_total_thickness
        self.similarity_method = similarity_method
        self.similarity_scale = similarity_scale
        self.exc_masks = exc_masks
        self.parcellation_name = parcellation_name
        self.correct_thickness_by_curvature = correct_thickness_by_curvature
        #> directory and filename (prefix which will be used for .npz and .jpg files)
        self.out_dir = out_dir
        self.filename = f'matrix_input-{self.input_type}'
        self.create()
        self.plot()
        self.save()

    def create(self):
        """
        Creates laminar thickness similarity matrix
        """
        print(f"""
        Creating similarity matrix:
            - input_type: {self.input_type},
            - parcellation_name: {self.parcellation_name},
            - exc_mask: {True if self.exc_masks else False}
        """)
        #> Reading laminar thickness or density
        print("Reading laminar input files")
        if self.input_type == 'thickness':
            self.laminar_data = helpers.read_laminar_thickness(
                exc_masks=self.exc_masks, 
                normalize_by_total_thickness=self.normalize_by_total_thickness, 
                regress_out_curvature=self.correct_thickness_by_curvature
            )
        elif self.input_type == 'density':
            self.laminar_data = helpers.read_laminar_density(
                exc_masks=self.exc_masks
            )
        #> create the similarity matrix
        if self.similarity_scale == 'parcel':
            self.matrix, self.valid_parcels = self.create_at_parcels()
        else:
            self.matrix, self.valid_parcels = self.create_at_vertices()

    def create_at_parcels(self):
        """
        Creates laminar thickness/density similarity matrix by taking Euclidean distance, Pearson's correlation
        or partial correltation (with the average laminar thickness pattern as the covariate) between
        average values of parcels

        Note: Partial correlation is based on "Using recursive formula" subsection in the wikipedia
        entry for "Partial correlation", which is also the same as Formula 2 in Paquola et al. PBio 2019
        (https://doi.org/10.1371/journal.pbio.3000284)
        Note 2: Euclidean distance is reversed (* -1) and rescaled to 0-1 (with 1 showing max similarity)

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                laminar thickness pattern
        """
        print("Concatenating and parcellating the data")
        self.parcellated_laminar_data = helpers.parcellate(
            self.laminar_data,
            self.parcellation_name,
            averaging_method='median'
            )
        #> concatenate left and right hemispheres
        self.concat_parcellated_laminar_data = helpers.concat_hemispheres(self.parcellated_laminar_data, dropna=True)
        print(f"Creating similarity matrix by {self.similarity_method} at parcel scale")
        #> Calculate parcel-wise similarity matrix
        if self.similarity_method in ['partial_corr', 'pearson']:
            if self.similarity_method == 'partial_corr':
                #> calculate partial correlation
                r_ij = np.corrcoef(self.concat_parcellated_laminar_data)
                mean_laminar_data = self.concat_parcellated_laminar_data.mean()
                r_ic = self.concat_parcellated_laminar_data\
                            .corrwith(mean_laminar_data, 
                            axis=1) # r_ic and r_jc are the same
                r_icjc = np.outer(r_ic, r_ic) # the second r_ic is actually r_jc
                matrix = (r_ij - r_icjc) / np.sqrt(np.outer((1-r_ic**2),(1-r_ic**2)))
            else:
                np.corrcoef(self.concat_parcellated_laminar_data.values)
            #> zero out correlations of 1 (to avoid division by 0)
            matrix[matrix==1] = 0
            #> Fisher's z-transformation
            matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
            #> zero out NaNs and inf
            matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        elif self.similarity_method == 'euclidean':
            #> calculate pair-wise euclidean distance
            matrix = sk.metrics.pairwise.euclidean_distances(self.concat_parcellated_laminar_data.values)
            #> make it negative (so higher = more similar) and rescale to range (0, 1)
            matrix = sk.preprocessing.minmax_scale(-matrix, (0, 1))
        #> determine valid parcels
        valid_parcels = self.concat_parcellated_laminar_data.index.tolist()
        return matrix, valid_parcels

    def create_at_vertices(self):
        """
        Creates laminar thickness/density similarity matrix by taking Euclidean distance or Pearson's correlation
        between all pairs of vertices between two pairs of parcels and then taking the average value of
        the resulting n_vert_parcel_i x n_vert_parcel_j matrix for the cell i, j of the parcels similarity matrix

        Note: Average euclidean distance is reversed (* -1) and rescaled to 0-1 (with 1 showing max similarity)

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                laminar data pattern
        """
        if self.similarity_method != 'euclidean':
            raise NotImplementedError("Correlation at vertex level is not implemented")
        #> Concatenate and parcellate the data
        print("Concatenating and parcellating the data")
        concat_laminar_data = np.concatenate([self.laminar_data['L'], self.laminar_data['R']], axis=0)
        self.concat_parcellated_laminar_data = helpers.parcellate(
            concat_laminar_data,
            self.parcellation_name,
            averaging_method=None
            ) # a groupby object
        n_parcels = len(self.concat_parcellated_laminar_data)
        #> Calculating similarity matrix
        print(f"Creating similarity matrix by {self.similarity_method} at vertex scale")
        matrix = pd.DataFrame(
            np.zeros((n_parcels, n_parcels)),
            columns = self.concat_parcellated_laminar_data.groups.keys(),
            index = self.concat_parcellated_laminar_data.groups.keys()
        )
        invalid_parcs = []
        for parc_i, vertices_i in self.concat_parcellated_laminar_data.groups.items():
            print("\tParcel", parc_i) # printing parcel_i name since it'll take a long time per parcel
            laminar_data_i = concat_laminar_data[vertices_i,:]
            laminar_data_i = laminar_data_i[~np.isnan(laminar_data_i).any(axis=1)]
            if laminar_data_i.shape[0] == 0: # sometimes all values may be NaN
                matrix.loc[parc_i, :] = np.NaN
                invalid_parcs.append(parc_i)
                continue
            for parc_j, vertices_j in self.concat_parcellated_laminar_data.groups.items():
                laminar_data_j = concat_laminar_data[vertices_j,:]
                laminar_data_j = laminar_data_j[~np.isnan(laminar_data_j).any(axis=1)]
                if laminar_data_i.shape[0] == 0:
                    matrix.loc[parc_i, :] = np.NaN
                else:
                    matrix.loc[parc_i, parc_j] = sk.metrics.pairwise.euclidean_distances(laminar_data_i, laminar_data_j).mean()
        #> make ED values negative (so higher = more similar) and rescale to range (0, 1)
        matrix = sk.preprocessing.minmax_scale(-matrix, (0, 1))
        #> store the valid parcels
        valid_parcels = sorted(list(set(self.concat_parcellated_laminar_data.groups.keys())-set(invalid_parcs)))
        return matrix, valid_parcels

    def save(self):
        """
        Save the matrix to a .npz file
        """
        np.savez_compressed(os.path.join(self.out_dir, self.filename)+'.npz', matrix=self.matrix)

    def plot(self):
        """
        Plot the matrix as heatmap
        """
        #> use different colors for different input types
        if self.input_type == 'thickness':
            cmap = 'RdBu_r'
        else:
            cmap = 'PiYG'
        # cmap = sns.color_palette("mako", as_cmap=True)
        fig, ax = plt.subplots(figsize=(7,7))
        sns.heatmap(
            self.matrix,
            vmin=np.quantile(self.matrix.flatten(),0.025),
            vmax=np.quantile(self.matrix.flatten(),0.975),
            cmap=cmap,
            cbar=False,
            ax=ax)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, self.filename)+'.png')
        clbar_fig = helpers.make_colorbar(
            vmin=np.quantile(self.matrix.flatten(),0.025),
            vmax=np.quantile(self.matrix.flatten(),0.975),
            cmap=cmap,
            orientation='vertical'
        )
        clbar_fig.savefig(os.path.join(self.out_dir, self.filename)+'_clbar.png')

#> laminar similarity gradients class
class LaminarSimilarityGradients:
    def __init__(self, gradient_input_type='thickness-density',
                 n_components=10, approach='dm',
                 kernel='normalized_angle', sparsity=0.9, **matrix_kwargs):
        """
        Initializes laminar similarity gradients based on LaminarSimilarityMatrix objects
        and the gradients fitting parameters

        Parameters
        ---------
        matrix_objs: (list) of LaminarSimilarityMatrix objects.
                     Usually len(matrix_objs) <= 2, but the code supports more matrices.
                     Matrices should have the same size and parcellation
        (the rest are brainspace.gradient.GradientMaps and LaminarSimilarityMatrix kwargs)

        Note: Multimodal gradient creation is based on Paquola 2020 PBio
        https://github.com/MICA-MNI/micaopen/tree/master/structural_manifold
        """        
        #> initialize variables
        self.n_components = n_components
        self.approach = approach
        self.kernel = kernel
        self.sparsity = sparsity
        self.gradient_input_type = gradient_input_type
        self._matrix_kwargs = matrix_kwargs
        #> create the directory
        os.makedirs(self.get_out_dir(), exist_ok=True)
        #> create matrcies
        self.matrix_objs = self.create_matrices(**matrix_kwargs)
        #> loading matrices +/- rank normalization
        if len(self.matrix_objs) > 1:
            print("Fusing matrices")
            self.matrices = self.rank_normalize()
            self.concat_matrix = np.hstack(self.matrices)
            self.plot_concat_matrix()
        else:
            self.concat_matrix = self.matrix_objs[0].matrix #misnomer
        #> create gradients
        print("Creating gradients")
        self.create()
        #> save the data
        print("Saving the gradients and plotting the scree plot")
        self.save()
        self.plot_scree()
        print(f"Gradients saved in {self.get_out_dir()}")

    def create_matrices(self, **matrix_kwargs):
        """
        Creates similarity matrix objects
        """
        matrix_objs = []
        for matrix_input_type in self.gradient_input_type.split('-'):
            matrix = LaminarSimilarityMatrix(
                input_type=matrix_input_type,
                out_dir=self.get_out_dir(),
                **matrix_kwargs)
            matrix_objs.append(matrix)
        return matrix_objs

    def rank_normalize(self):
        """
        Perform rank normalization of input matrices. Also rescales
        matrices[1:] based on matrics[0].

        Based on Paquola 2020 PBio
        https://github.com/MICA-MNI/micaopen/tree/master/structural_manifold
        Note: This function does not zero out negative or NaNs as this has
              already been done in LaminarThicknessSimilarityMatrix.create
        """
        rank_normalized_matrices = []
        #> rank normalize the first matrix
        rank_normalized_matrices.append(
            scipy.stats.rankdata(self.matrix_objs[0].matrix)
            .reshape(self.matrix_objs[0].matrix.shape)
        )
        #> rank normalize and rescale next matrices
        for matrix_obj in self.matrix_objs[1:]:
            #> rank normalize (and flatten) the matrix
            rank_normalized_matrix_flat = scipy.stats.rankdata(matrix_obj.matrix)
            #> rescale it by the first matrix
            rank_normalized_rescaled_matrix_flat = np.interp(
                rank_normalized_matrix_flat,
                (rank_normalized_matrix_flat.min(), rank_normalized_matrix_flat.max()),
                (rank_normalized_matrices[0].min(), rank_normalized_matrices[0].max())
            )
            rank_normalized_matrices.append(
                rank_normalized_rescaled_matrix_flat.reshape(matrix_obj.matrix.shape)
            )
        return rank_normalized_matrices

    def create(self):
        """
        Creates the GradientMaps object and fits the data
        """
        self.gm = brainspace.gradient.GradientMaps(
            n_components=self.n_components, 
            approach=self.approach, 
            kernel=self.kernel, 
            random_state=912)
        self.gm.fit(self.concat_matrix, sparsity=self.sparsity)

    def project_to_surface(self):
        """
        Project the gradients on BigBrain surface

        Returns
        -------
        gradient_maps: (np.ndarray) n_vertices [both hemispheres] x n_gradients
        """
        #> load concatenated parcellation map
        concat_parcellation_map = helpers.load_parcellation_map(self.matrix_objs[0].parcellation_name, concatenate=True)
        #>> create a gradients dataframe including all parcels, where invalid parcels are NaN
        #   (this is necessary to be able to project it to the parcellation)
        gradients_df = pd.concat(
            [
                pd.DataFrame(
                    self.gm.gradients_, 
                    index=self.matrix_objs[0].valid_parcels # valid parcels
                    ),
                pd.Series(np.unique(concat_parcellation_map), name='parcels') # all parcels
            ], axis=1).set_index('parcels')
        #> get the map of gradients by indexing at parcellation labels
        gradient_maps = gradients_df.loc[concat_parcellation_map].values # shape: vertex X gradient
        return gradient_maps

    def get_out_dir(self):
        """
        Get the path to the directory of the gradient (and create it if needed)
        """
        out_dir = f'gradient_input-{self.gradient_input_type}_parc-{self._matrix_kwargs["parcellation_name"]}_approach-{self.approach}'
        if self._matrix_kwargs['exc_masks']:
            out_dir += '_excmask-adys'
        if self._matrix_kwargs['correct_thickness_by_curvature']:
            out_dir += '_corr-curv'
        out_dir = out_dir.lower()
        out_dir = os.path.join(DATA_DIR, 'gradient', out_dir)
        return out_dir

    def save(self):
        """
        Save the gradients map projected on surface and lambdas
        """
        np.savez_compressed(
            os.path.join(self.get_out_dir(),'gradients_surface.npz'),
            surface=self.project_to_surface()
        )
        np.savetxt(
            os.path.join(self.get_out_dir(),'lambdas.txt'),
            self.gm.lambdas_
        )

    def plot_scree(self):
        """
        Save the scree plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        sns.scatterplot(
            x = np.arange(1, self.gm.lambdas_.shape[0]+1).astype('str'),
            y = (self.gm.lambdas_ / self.gm.lambdas_.sum()),
            ax=axes[0]
            )
        axes[0].set_title(f'Variance explained by each gradient\n(relative to the total variance in the first {self.gm.lambdas_.shape[0]} gradients)')
        sns.lineplot(
            x = np.arange(1, self.gm.lambdas_.shape[0]+1).astype('str'),
            y = np.cumsum(self.gm.lambdas_) / self.gm.lambdas_.sum(), 
            ax=axes[1]
            )
        axes[1].set_title(f'Cumulative variance explained by the gradients\n(out of total variance in the first {self.gm.lambdas_.shape[0]} gradients)')
        for ax in axes:
            ax.set_xlabel('Gradient')
        fig.savefig(
            os.path.join(self.get_out_dir(),'scree.png'),
            dpi=192)

    def plot_concat_matrix(self):
        """
        Plot the concatenated matrix as heatmap
        """
        fig, ax = plt.subplots(figsize=(7*len(self.matrix_objs),7))
        sns.heatmap(
            self.concat_matrix,
            vmax=np.quantile(self.concat_matrix.flatten(),0.75),
            cbar=False,
            ax=ax)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.get_out_dir(),f'matrix_input-{self.gradient_input_type}.png'), 
            dpi=192)


#> Create several gradients with different options
#  The first loop is the main analysis and the rest are done for assessing robustness
for gradient_input_type in ['thickness']: #['thickness', 'thickness-density']:
    for parcellation_name in ['sjh']:
        for exc_masks in [None, adysgranular_masks]:
            for correct_thickness_by_curvature in [False, True]:
                gradients = LaminarSimilarityGradients(
                    gradient_input_type=gradient_input_type,
                    parcellation_name=parcellation_name,
                    exc_masks=exc_masks,
                    correct_thickness_by_curvature=correct_thickness_by_curvature)