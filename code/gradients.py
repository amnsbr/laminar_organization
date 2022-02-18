
import os
import numpy as np
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm # color maps
import brainspace.gradient

import helpers
import datasets

class MicrostructuralCovarianceGradients:
    """
    Creates and plots gradients of microstructural covariance
    """
    def __init__(self, matrix_obj, n_components_create=10, n_components_report=2,
                 approach='dm', kernel='normalized_angle', sparsity=0.9, plot_surface=False):
        """
        Initializes laminar similarity gradients based on LaminarSimilarityMatrix objects
        and the gradients fitting parameters

        Parameters
        ---------
        matrix: (MicrostructuralCovarianceMatrix)
        n_components_create: (int) number of components to calculate
        n_components_report: (int) number of first n components to report / plot / associate
        approach: (str) dimensionality reduction approach
        kernel: (str) affinity matrix calculation kernel
        sparsity: (int) proportion of smallest elements to zero-out for each row.
            For bimodal matrix it will be adjusted to 1-((1-sparsity)*2)
        plot_surface: (bool) Default: False
        """
        self.matrix_obj = matrix_obj
        self._n_components_create = n_components_create
        self._n_components_report = n_components_report #TODO: determine this programmatically based on scree plot
        self._approach = approach
        self._kernel = kernel
        # determine sparsity based on number of matrices
        self.n_parcels = self.matrix_obj.matrix.shape[0]
        self.n_matrices = self.matrix_obj.matrix.shape[1] // self.n_parcels
        self._sparsity = 1 - ((1 - sparsity) * self.n_matrices)
        # directory
        self.dir_path = self._get_dir_path()
        os.makedirs(self.dir_path, exist_ok=True)
        if os.path.exists(os.path.join(self.dir_path, 'gradients_surface.npz')):
            self._load()
        else:
            print(f"Creating gradients in {self.dir_path}")
            self._create()
            self._save()
            if plot_surface:
                self.plot_surface()
            self.plot_scatter()
            self.plot_scree()
            self.plot_reordered_matrix()
            self.plot_binned_profile()

    def _create(self):
        """
        Creates the GradientMaps object and fits the data
        """
        #> initialize GradientMaps object and fit it to data
        self.gm = brainspace.gradient.GradientMaps(
            n_components=self._n_components_create, 
            approach=self._approach, 
            kernel=self._kernel, 
            random_state=912
            )
        self.gm.fit(self.matrix_obj.matrix.values, sparsity=self._sparsity)
        self.lambdas = self.gm.lambdas_
        #> add parcel labels
        self.labeled_gradients = pd.DataFrame(
            self.gm.gradients_,
            index=self.matrix_obj.matrix.index
            )
        #> project to surface
        self.gradients_surface = helpers.deparcellate(
            self.labeled_gradients,
            self.matrix_obj.parcellation_name
            )
        
    def _get_dir_path(self):
        """
        Get path to the directory of gradients which would be
        <path-to-matrix>/<gradient-options>
        """
        parent_dir = self.matrix_obj.dir_path
        sub_dir = f'gradients_approach-{self._approach}'\
            + f'_kernel-{self._kernel}'\
            + f'_sparsity-{self._sparsity}'.replace('.','')\
            + f'_n-{self._n_components_create}'
        return os.path.join(parent_dir, sub_dir)

    def _save(self):
        """
        Save the labeled gradients as .csv, surface maps as .npz, and
        lambdas as .txt
        """
        self.labeled_gradients.to_csv(
            os.path.join(self.dir_path, 'gradients_parcels.csv'), 
            index_label='parcel'
            )
        np.savez_compressed(
            os.path.join(self.dir_path,'gradients_surface.npz'),
            surface=self.gradients_surface
            )
        np.savetxt(
            os.path.join(self.dir_path,'lambdas.txt'),
            self.lambdas
            )

    def _load(self):
        """
        Load labeled gradients, surface map
        """
        self.labeled_gradients = pd.read_csv(
            os.path.join(self.dir_path, 'gradients_parcels.csv'),
            index_col='parcel')
        self.labeled_gradients.columns = self.labeled_gradients.columns.map(int)
        self.gradients_surface = np.load(os.path.join(self.dir_path, 'gradients_surface.npz'))['surface']
        self.lambdas = np.loadtxt(os.path.join(self.dir_path, 'lambdas.txt'))

    def plot_surface(self, layout='grid', inflate=False):
        """
        Plots the first `n_gradients` of `gradient_file`
        Note: It is computationally intensive (too many vertices)
        """
        for gradient_num in range(1, self._n_components_report+1):
            helpers.plot_on_bigbrain_nl(
                self.gradients_surface[:, gradient_num-1],
                filename=os.path.join(self.dir_path, f'surface_{layout}_G{gradient_num}.png'),
                layout=layout,
                inflate=inflate,
            )

    def plot_scatter(self, remove_ticks=True):
        """
        Plot scatter plot of gradient values for G1 (x-axis) and G2 (y-axis) with
        colors representing G3

        Parameters
        ----------
        remove_ticks: (bool) remove ticks so that colorbars can replace them (manually)
        """
        fig, ax = plt.subplots(figsize=(6,5))
        ax = sns.scatterplot(
            data=self.labeled_gradients, 
            x=0, # G1
            y=1, # G2
            hue=2, # G3
            palette=cmcrameri.cm.lajolla_r, # G3 cmap
            legend=False, ax=ax)
        ax.set_xlabel('G1')
        ax.set_ylabel('G2')
        if remove_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.savefig(
            os.path.join(self.dir_path,f'scatter.png'), 
            dpi=192)

    def plot_scree(self, normalize=False):
        """
        Plot the lamdas

        Parameters
        ---------
        normalize: (bool) normalize the lambdas by sum
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(1, self.lambdas.shape[0]+1).astype('str')
        if normalize:
            y = (self.lambdas / self.lambdas.sum())
        else:
            y = self.lambdas
        ax.plot(
            x, y,
            marker = 'o', 
            linestyle = 'dashed',
            linewidth = 0.5,
            markersize = 2,
            color = 'black',
            )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(range(1, 11))
        fig.savefig(
            os.path.join(self.dir_path,f'scree{"_normalized" if normalize else ""}.png'),
            dpi=192
            )
        #> cumulative variance explained
        fig, ax = plt.subplots(figsize=(6,4))
        y = (self.lambdas.cumsum() / self.lambdas.sum()) * 100
        ax.plot(
            x, y,
            linewidth = 0.5,
            color = 'grey',
            )
        fig.savefig(
            os.path.join(self.dir_path,'scree_cum.png'),
            dpi=192
            )

    def plot_reordered_matrix(self):
        """
        Plot the input matrix reordered by gradient values
        """
        #> use different colors for different input types
        if self.matrix_obj.input_type == 'thickness':
            cmap = sns.color_palette("RdBu_r", as_cmap=True)
        elif self.matrix_obj.input_type == 'density':
            cmap = sns.color_palette("rocket", as_cmap=True)
        else:
            cmap = sns.color_palette("RdBu_r", as_cmap=True)
        if self.n_matrices == 1:
            #> in unimodal case add ordered matrices by the first 3 gradients
            for g_idx in range(self._n_components_report):
                #> sort parcels by gradient values
                sorted_parcels = self.labeled_gradients.sort_values(by=g_idx).index
                #> reorder matrix by sorted parcels
                reordered_matrix = self.matrix_obj.matrix.loc[sorted_parcels, sorted_parcels].values
                helpers.plot_matrix(
                    reordered_matrix,
                    os.path.join(self.dir_path, f'matrix_order-G{g_idx+1}'),
                    cmap=cmap,
                    )
        else:
            for g_idx in range(self._n_components_report):
                #> sort parcels by gradient values
                sorted_parcels = self.labeled_gradients.sort_values(by=g_idx).index
                #> split the matrix to square matrices, reorder each square matrix
                #  separately, and then hstack reordered square matrices and plot it
                reordered_split_matrices = []
                for i in range(self.n_matrices):
                    split_matrix = self.matrix_obj.matrix.iloc[:, self.n_parcels*i : self.n_parcels*(i+1)]
                    reordered_matrix = split_matrix.loc[sorted_parcels, sorted_parcels].values
                    reordered_split_matrices.append(reordered_matrix)
                reordered_split_matrices = np.hstack(reordered_split_matrices)
                helpers.plot_matrix(
                    reordered_split_matrices,
                    os.path.join(self.dir_path, f'matrix_order-G{g_idx+1}'),
                    cmap=cmap,
                    )

    def plot_binned_profile(self, n_bins=10, cmap='Blues'):
        """
        Plots the relative laminar thickness (TODO: and density) of `n_bins` bins of the top gradients
        """
        if self.matrix_obj.input_type != 'thickness':
            print(f"Plotting binned profiles is not implemented for input {self.matrix_obj.input_type}")
            return
        #> loading and parcellating the laminar thickness
        laminar_thickness = self.matrix_obj._load_input_data()
        parcellated_laminar_thickness = helpers.parcellate(laminar_thickness, self.matrix_obj.parcellation_name)
        parcellated_laminar_thickness = helpers.concat_hemispheres(parcellated_laminar_thickness, dropna=True)
        # re-normalize small deviations from sum=1 because of parcellation
        parcellated_laminar_thickness = parcellated_laminar_thickness.divide(parcellated_laminar_thickness.sum(axis=1), axis=0)
        for gradient_num in range(1, self._n_components_report+1):
            binned_parcels_laminar_thickness = parcellated_laminar_thickness.copy()
            binned_parcels_laminar_thickness['bin'] = pd.cut(self.labeled_gradients[gradient_num-1], 10)
            #> calculate average laminar thickness at each bin
            bins_laminar_thickness = binned_parcels_laminar_thickness.groupby('bin').mean().reset_index(drop=True)
            bins_laminar_thickness.columns = [f'Layer {idx+1}' for idx in range(6)]
            #> reverse the columns so that in the plot Layer 6 is at the bottom
            bins_laminar_thickness = bins_laminar_thickness[bins_laminar_thickness.columns[::-1]]
            #> normalize to sum of 1 at each bin
            bins_laminar_thickness = bins_laminar_thickness.divide(bins_laminar_thickness.sum(axis=1), axis=0)
            #> plot the relative thickness of layers 6 to 1
            # TODO: combine this with misc.py/plot_parcels_laminar_profile and put it in helpers.py
            # TODO: use BigBrain colormap
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                x = bins_laminar_thickness.index,
                height = bins_laminar_thickness['Layer 6'],
                width = 0.95,
                color=plt.cm.get_cmap(cmap)(6/6),
                )
            for layer_num in range(5, 0, -1):
                ax.bar(
                    x = bins_laminar_thickness.index,
                    height = bins_laminar_thickness[f'Layer {layer_num}'],
                    width = 0.95,
                    bottom = bins_laminar_thickness.cumsum(axis=1)[f'Layer {layer_num+1}'],
                    color=plt.cm.get_cmap(cmap)(layer_num/6),
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f'G{gradient_num} bins')
            ax.set_ylabel('Relative laminar thickness')
            for _, spine in ax.spines.items():
                spine.set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.dir_path, f'binned_profile_G{gradient_num}.png'), dpi=192)
            clfig = helpers.make_colorbar(
                self.labeled_gradients[gradient_num-1].min(), 
                self.labeled_gradients[gradient_num-1].max(),
                bins=10, 
                orientation='horizontal', figsize=(6,4))
            clfig.savefig(os.path.join(self.dir_path, f'binned_profile_G{gradient_num}_clbar.png'), dpi=192)
