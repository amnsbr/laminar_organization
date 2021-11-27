import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from seaborn import heatmap, scatterplot, lineplot
import nilearn.surface
import brainspace.gradient

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
    def __init__(self, input_type='thickness', normalize_by_total_thickness=True, 
                 exc_masks=None,  parcellation_name='sjh', 
                 averaging_method='median', similarity_method='partial_corr', 
                 plot=False, filename=None):
        """
        Initializes laminar thickness similarity matrix object

        Parameters
        ---------
        input_type: (str) Type of input laminar data
            - 'thickness' (default)
            - 'density'
        normalize: (bool) Normalize by total thickness. Default: True
        exc_masks: (dict of str) Path to the surface masks of vertices that should be excluded (L and R) (format: .npy)
        parcellation_name: (str) Parcellation scheme
            - 'sjh'
        averaging_method: (str) Method of averaging over vertices within a parcel. 
            - 'median'
            - 'mean'
        similarity_method: (str) 
            - 'partial_corr' -> partial correlation, 
            - 'pearson' -> Pearson's correlation
            - 'KL' -> KL-divergence of thickness distribution between parcels
        plot: (bool) Plot the similarity matrix. Default: False
        filenmae: (str) Output path to the figure of matrix heatmap
        """
        #> save parameters as class fields
        self.input_type = input_type
        self.normalize_by_total_thickness = normalize_by_total_thickness
        self.exc_masks = exc_masks
        self.parcellation_name = parcellation_name
        self.averaging_method = averaging_method
        self.similarity_method = similarity_method
        #> get the number of vertices in each hemisphere
        self.n_hem_vertices = np.loadtxt(
            os.path.join(
                DATA_DIR, 'surface',
                'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
                )).size
        print(f"""
        Creating similarity matrix:
            - input_type: {input_type},
            - parcellation_name: {parcellation_name},
            - exc_mask: {True if exc_masks else False}
        """)
        self.create()
        self.plot()
        self.save()
        print(f"Matrix saved in {self.get_path()}")

    def create(self):
        """
        Creates laminar thickness similarity matrix
        """
        print("Reading laminar input files")
        #> Reading laminar thickness or density
        if self.input_type == 'thickness':
            self.laminar_data = self.read_laminar_thickness()
        elif self.input_type == 'density':
            self.laminar_data = self.read_laminar_density()
        #> Create the similarity matrix
        if self.similarity_method != 'KL':
            #> Parcellate the data
            print("Parcellating the data")
            self.parcellated_laminar_data = self.parcellate(self.laminar_data)
            #> Calculate parcel-wise full or partial correlation
            print(f"Creating similarity matrix by {self.similarity_method}")
            self.matrix = self.create_by_corr()
        else: #TODO: KL-divergence
            raise NotImplementedError

    def read_laminar_thickness(self):
        """
        Reads laminar thickness data from 'data' folder and after masking out
        `exc_mask` returns 6-d laminar thickness arrays for left and right hemispheres

        Retruns
        --------
        laminar_thickness: (dict of np.ndarray) 6 x n_vertices for laminar thickness of L and R hemispheres
        """
        #> get the number of vertices
        laminar_thickness = {}
        for hem in ['L', 'R']:
            #> read the laminar thickness data from bigbrainwrap .txt files
            laminar_thickness[hem] = np.empty((6, self.n_hem_vertices))
            for layer_num in range(1, 7):
                laminar_thickness[hem][layer_num-1, :] = np.loadtxt(
                    os.path.join(
                        DATA_DIR, 'surface',
                        f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
                        ))
            #> remove the exc_mask
            if self.exc_masks:
                exc_mask_map = np.load(os.path.join(
                    DATA_DIR, 'surface',
                    self.exc_masks[hem]
                    ))
                laminar_thickness[hem][:, exc_mask_map] = np.NaN
        return laminar_thickness

    def read_laminar_density(self, method='mean'):
        """
        Reads laminar density data from 'src' folder, takes the average of sample densities
        for each layer, and after masking out `exc_mask` returns 6-d average laminar density 
        arrays for left and right hemispheres

        Parameters
        ----------
        method: (str)
            - mean
            - median

        Retruns
        --------
        laminar_density: (dict of np.ndarray) 6 x n_vertices for laminar density of L and R hemispheres
        """
        laminar_density = {}
        for hem in ['L', 'R']:
            #> read the laminar thickness data from bigbrainwrap .txt files
            laminar_density[hem] = np.empty((6, self.n_hem_vertices))
            for layer_num in range(1, 7):
                profiles = np.load(
                    os.path.join(
                        DATA_DIR, 'surface',
                        f'tpl-bigbrain_hemi-{hem[0].upper()}_desc-layer-{layer_num}_profiles_nsurf-10.npz'
                        ))['profiles']
                if method == 'mean':
                    laminar_density[hem][layer_num-1, :] = profiles.mean(axis=0)
                elif method == 'median':
                    laminar_density[hem][layer_num-1, :] = np.median(profiles, axis=0)
            #> remove the exc_mask
            if self.exc_masks:
                exc_mask_map = np.load(os.path.join(
                    DATA_DIR, 'surface',
                    self.exc_masks[hem]
                    ))
                laminar_density[hem][:, exc_mask_map] = np.NaN
        return laminar_density

    def parcellate(self, surface_data):
        """
        Parcellates `surface data` using `parcellation` and by taking the
        median or mean (specified via `averaging_method`) of the vertices within each parcel.

        Parameters
        ----------
        surface_data: (dict of np.ndarray) p x n_vertices surface data of L and R hemispheres

        Returns
        ---------
        parcellated_data: (dict of pd.DataFrame) n_parcels x 6 for laminar data of L and R hemispheres
        """
        parcellated_data = {}
        for hem in ['L', 'R']:
            parcellation_map = nilearn.surface.load_surf_data(
                os.path.join(
                    DATA_DIR, 'parcellation', 
                    f'tpl-bigbrain_hemi-{hem}_desc-{self.parcellation_name}_parcellation.label.gii')
                )
            parellated_vertices = (
                pd.DataFrame(surface_data[hem].T, index=parcellation_map)
                .reset_index()
                .groupby('index')
            )
            if self.averaging_method == 'median':
                parcellated_data[hem] = parellated_vertices.median()
            elif self.averaging_method == 'mean':
                parcellated_data[hem] = parellated_vertices.mean()
            
        return parcellated_data

    def concat_hemispheres(self, parcellated_data, dropna=True):
        """
        Concatenates the parcellated data of L and R hemispheres

        Parameters
        ----------
        parcellated_data: (dict of pd.DataFrame) n_parcels x 6 for laminar data of L and R hemispheres

        Returns
        ----------
        concat_data: (pd.DataFrame) n_parcels*2 x 6
        """
        #> create a deep copy of R data since its index is going
        #  to be (temporarily) altered
        parcellated_data_copy = {
            'L': parcellated_data['L'],
            'R': parcellated_data['R'].copy(deep=True)
        }
        #> make R index continuous to L index
        parcellated_data_copy['R'].index = \
            parcellated_data_copy['L'].index[-1] \
            + 1 \
            + parcellated_data_copy['R'].index
        #> concatenate hemispheres and drop NaN if needed
        concat_data = pd.concat(
            [
                parcellated_data_copy['L'], 
                parcellated_data_copy['R']
            ],
            axis=0)
        if dropna:
            concat_data = concat_data.dropna()
        return concat_data


    def create_by_corr(self):
        """
        Creates laminar thickness similarity matrix by taking partial or Pearson's correlation
        between pairs of parcels. In partial correlation the average laminar thickness pattern 
        is set as the covariate

        Note: Partial correlation is based on "Using recursive formula" subsection in the wikipedia
        entry for "Partial correlation", which is also the same as Formula 2 in Paquola et al. PBio 2019
        (https://doi.org/10.1371/journal.pbio.3000284)

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                laminar thickness pattern
        """
        #> normalize by total thickness if needed [TODO: also by total density?]
        if (self.input_type == ['thickness']) and self.normalize_by_total_thickness:
            for hem in ['L', 'R']:
                self.parcellated_laminar_data[hem] =\
                    self.parcellated_laminar_data[hem].divide(
                        self.parcellated_laminar_data[hem].sum(axis=1), axis=0
                        )
        #> concatenate left and right hemispheres
        concat_parcellated_laminar_data = self.concat_hemispheres(self.parcellated_laminar_data, dropna=True)
        #> create similarity matrix
        if self.similarity_method == 'partial_corr':
            #> calculate partial correlation
            r_ij = np.corrcoef(concat_parcellated_laminar_data)
            mean_laminar_data = concat_parcellated_laminar_data.mean()
            r_ic = concat_parcellated_laminar_data\
                        .corrwith(mean_laminar_data, 
                        axis=1) # r_ic and r_jc are the same
            r_icjc = np.outer(r_ic, r_ic) # the second r_ic is actually r_jc
            matrix = (r_ij - r_icjc) / np.sqrt(np.outer((1-r_ic**2),(1-r_ic**2)))
            #> zero out negative values
            matrix[matrix<0] = 0
            #> zero out correlations of 1 (to avoid division by 0)
            matrix[matrix==1] = 0
            #> Fisher's z-transformation
            matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
            #> zero out NaNs and inf
            matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        elif self.similarity_method == 'pearson':
            matrix = np.corrcoef(concat_parcellated_laminar_data)
            #> zero out negative values
            matrix[matrix < 0] = 0
            #> zero out correlations of 1 (to avoid division by 0)
            matrix[matrix==1] = 0
            #> Fisher's z-transformation
            matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
            #> zero out NaNs and inf
            matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        return matrix
    
    def get_path(self):
        outfilename = f'matrix_input-{self.input_type}_simmethod-{self.similarity_method}_parc-{self.parcellation_name}_avgmethod-{self.averaging_method}'
        if self.normalize_by_total_thickness:
            outfilename += '_normalized'
        if self.exc_masks: #TODO specify the name of excmask
            outfilename += '_excmask'
        outfilename = outfilename.lower()
        return os.path.join(DATA_DIR, 'matrix', outfilename)

    def save(self, outfile=None):
        """
        Save the matrix to a .npz file
        """
        if not outfile:
            outfile = self.get_path()
        np.savez_compressed(outfile, matrix=self.matrix)

    def plot(self, outfile=None):
        """
        Plot the matrix as heatmap
        """
        fig, ax = plt.subplots(figsize=(7,7))
        heatmap(
            self.matrix,
            vmax=np.quantile(self.matrix.flatten(),0.75),
            cbar=False,
            ax=ax)
        ax.axis('off')
        if not outfile:
            outfile = self.get_path() + '.png'
        fig.tight_layout()
        fig.savefig(outfile)

#> laminar similarity gradients class
class LaminarSimilarityGradients:
    def __init__(self, matrix_objs, do_rank_normalization=True, 
                 n_components=10, approach='dm',
                 kernel='normalized_angle', sparsity=0.9,):
        """
        Initializes laminar similarity gradients based on LaminarSimilarityMatrix objects
        and the gradients fitting parameters

        Parameters
        ---------
        matrix_objs: (list) of LaminarSimilarityMatrix objects.
                     Usually len(matrix_objs) <= 2, but the code supports more matrices.
                     Matrices should have the same size and parcellation
        rank_normalize: (bool) Peform rank normalization on the matrices. Default: True
        (the rest are brainspace.gradient.GradientMaps kwargs)

        Note: Multimodal gradient creation is based on Paquola 2020 PBio
        https://github.com/MICA-MNI/micaopen/tree/master/structural_manifold
        """        
        #> initialize variables
        self.matrix_objs = matrix_objs
        self.multimodal = len(self.matrix_objs) > 1
        self.do_rank_normalization = do_rank_normalization
        self.n_components = n_components
        self.approach = approach
        self.kernel = kernel
        self.sparsity = sparsity
        #> loading matrices +/- rank normalization
        if self.do_rank_normalization:
            print("Rank normalization")
            self.matrices = self.rank_normalize()
        else:
            self.matrices = [matrix_obj.matrix for matrix_obj in self.matrix_objs]
        #> concatenate matrices horizontally
        self.concat_matrix = np.hstack(self.matrices)
        #> create gradients
        print("Creating gradients")
        self.create()
        #> save the data
        print("Saving the gradients and plotting the scree plot")
        self.save_surface()
        self.save_lambdas()
        self.plot_scree()
        self.plot_concat_matrix()
        print(f"Gradients saved in {self.get_path()}")

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
        """
        #> load the parcellation map
        parcellation_maps = {}
        for hem in ['L', 'R']:
            parcellation_maps[hem] = nilearn.surface.load_surf_data(
                os.path.join(
                    DATA_DIR, 'parcellation', 
                    f'tpl-bigbrain_hemi-{hem}_desc-{self.matrix_objs[0].parcellation_name}_parcellation.label.gii')
                )
        #> relabel right hemisphere in continuation with the labels from left hemisphere
        parcellation_maps['R'] = parcellation_maps['L'].max() + 1 + parcellation_maps['R']
        #> concatenate left and right hemispheres
        concat_parcellation_map = np.concatenate([parcellation_maps['L'], parcellation_maps['R']])
        #> project the gradient values on the surface
        #> add back the masked out parcels and set their gradient values as NaN
        #  (there should be easier solutions but this was a simple method to do it in one line)
        # TODO: make this more readable/understandable
        #>> load parcellated laminar data (we only need the index for valid non-NaN parcels)
        concat_parcellated_laminar_data = self.matrix_objs[0].concat_hemispheres(self.matrix_objs[0].parcellated_laminar_data, dropna=False)
        #>> create a gradients dataframe including all parcels, where invalid parcels are NaN
        #   (this is necessary to be able to project it to the parcellation)
        gradients_df = pd.concat(
            [
                pd.DataFrame(
                    self.gm.gradients_, 
                    index=concat_parcellated_laminar_data.dropna().index # valid parcels
                    ),
                concat_parcellated_laminar_data.index.to_series() # all parcels
            ], axis=1).drop(columns=['index'])
        #> get the map of gradients by indexing at parcellation labels
        gradient_maps = gradients_df.loc[concat_parcellation_map].values # shape: vertex X gradient
        return gradient_maps

    def get_path(self):
        path = self.matrix_objs[0].get_path()\
            .replace('matrix', 'gradient')\
            .replace(
                'input-', 
                f'input-{"".join(m.input_type for m in self.matrix_objs[1:])}')\
            + f'_gapproach-{self.gm.approach}'
        if self.do_rank_normalization:
            path += '_ranknormalized'
        return path

    def save_surface(self):
        """
        Save the gradients map projected on surface
        """
        np.savez_compressed(
            self.get_path()+'_surface',
            surface=self.project_to_surface()
        )

    def save_lambdas(self):
        """
        Save lambdas as txt
        """
        np.savetxt(
            self.get_path()+'_lambdas.txt',
            self.gm.lambdas_
        )

    def plot_scree(self):
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        scatterplot(
            x = np.arange(1, self.gm.lambdas_.shape[0]+1).astype('str'),
            y = (self.gm.lambdas_ / self.gm.lambdas_.sum()),
            ax=axes[0]
            )
        axes[0].set_title(f'Variance explained by each gradient\n(relative to the total variance in the first {self.gm.lambdas_.shape[0]} gradients)')
        lineplot(
            x = np.arange(1, self.gm.lambdas_.shape[0]+1).astype('str'),
            y = np.cumsum(self.gm.lambdas_) / self.gm.lambdas_.sum(), 
            ax=axes[1]
            )
        axes[1].set_title(f'Cumulative variance explained by the gradients\n(out of total variance in the first {self.gm.lambdas_.shape[0]} gradients)')
        for ax in axes:
            ax.set_xlabel('Gradient')
        fig.savefig(self.get_path()+'_scree.png', dpi=192)

    def plot_concat_matrix(self, outfile=None):
        """
        Plot the matrix as heatmap
        """
        fig, ax = plt.subplots(figsize=(7*len(self.matrix_objs),7))
        heatmap(
            self.concat_matrix,
            vmax=np.quantile(self.concat_matrix.flatten(),0.75),
            cbar=False,
            ax=ax)
        ax.axis('off')
        if not outfile:
            outfile = self.get_path() + '_concatmat.png'
        fig.tight_layout()
        fig.savefig(outfile)


#> Create several gradients with different options
#  The first loop is the main analysis and the rest are done for assessing robustness
for gradient_input_type in ['thickness_density', 'thickness', 'density']:
    for parcellation_name in ['sjh']:
        for exc_masks in [adysgranular_masks, None]:
            matrices = []
            for matrix_input_type in gradient_input_type.split('_'):
                matrix = LaminarSimilarityMatrix(
                    input_type=matrix_input_type, 
                    parcellation_name=parcellation_name,
                    exc_masks=exc_masks)
                matrices.append(matrix)
            gradients = LaminarSimilarityGradients(matrices)