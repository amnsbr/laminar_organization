
import os
import itertools
import numpy as np
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm # color maps
import brainspace.gradient
import scipy.stats
import ptitprince
import nibabel
from dominance_analysis import Dominance

import helpers
import datasets
import matrices

#> specify directories
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')


class ContCorticalSurface:
    """
    Generic class for common functions on cortical surfaces with continuous data
    """
    def correlate(self, other, parcellation_name=None, n_perm=1000,
                 x_columns=None, y_columns=None):
        """
        Calculate the correlation of surface maps with spin test
        and plot the scatter plots

        Parameters
        ---------
        self, other (CorticalSurface): with surf_data fields which are n_vert x n_features np.ndarrays
        parcellation_name (str): used for the regression plot
        n_perm (int): number of permutations for spin test. Max: 1000
        x_columns, y_columns (list of str): selected columns from self (x) and other (y)
        """
        out_dir = os.path.join(self.dir_path, f'correlation_{other.label.lower().replace(" ", "_")}')
        os.makedirs(out_dir, exist_ok=True)
        #> select columns
        if not x_columns:
            x_columns = self.columns
        if not y_columns:
            y_columns = other.columns
        x_data = pd.DataFrame(self.surf_data, columns=self.columns).loc[:, x_columns]
        y_data = pd.DataFrame(other.surf_data, columns=other.columns).loc[:, y_columns]
        #> spin test after downsampling
        print("Calculating correlations with spin test")
        coefs, pvals, coefs_null_dist =  helpers.spin_test(
            surface_data_to_spin = helpers.downsample(x_data.values), 
            surface_data_target = helpers.downsample(y_data.values),
            n_perm=n_perm,
            is_downsampled=True
            )
        #> save null distribution for future reference
        np.savez_compressed(
            os.path.join(out_dir, 'coefs_null_dist.npz'),
            coefs_null_dist=coefs_null_dist
        )
        #> clean and save the results
        coefs = pd.DataFrame(coefs)
        pvals = pd.DataFrame(pvals)
        coefs.columns = pvals.columns = x_columns
        coefs.index = pvals.index = y_columns
        coefs.to_csv(os.path.join(out_dir, 'coefs.csv'))
        pvals.to_csv(os.path.join(out_dir, 'pvals.csv'))
        #> regression plots
        if not parcellation_name:
            x_parcellated = pd.DataFrame(helpers.downsample(x_data.values))
            y_parcellated = pd.DataFrame(helpers.downsample(y_data.values))
        else:
            x_parcellated = helpers.parcellate(x_data.values, parcellation_name)
            y_parcellated = helpers.parcellate(y_data.values, parcellation_name)
        x_parcellated.columns = x_columns
        y_parcellated.columns = y_columns
        for x_column in x_columns:
            for y_column in y_columns:
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.regplot(
                    x=x_parcellated.loc[:,x_column], 
                    y=y_parcellated.loc[:,y_column],
                    scatter_kws=dict(alpha=0.2, s=5, color='grey'),
                    line_kws=dict(color='red'),
                    ax=ax)
                sns.despine(offset=10, trim=True, ax=ax)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                #> add correlation coefficients and p vals on the figure
                text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
                text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
                ax.text(text_x, text_y, 
                        f'r = {coefs.loc[y_column, x_column]:.2f}; $\mathregular{{p_{{spin}}}}$ = {pvals.loc[y_column, x_column]:.2f}',
                        color='black',
                        size=8,
                        multialignment='left')
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        out_dir, 
                        f'{x_column.lower().replace(" ", "_")}-'\
                        + f'{y_column.lower().replace(" ", "_")}.png'),
                    dpi=192
                    )

class Gradients(ContCorticalSurface):
    """
    Generic class for creating and plotting gradients
    """
    def __init__(self, matrix_obj, n_components_create=10, n_components_report=2,
                 approach='dm', kernel='normalized_angle', sparsity=0.9, fair_sparsity=True,
                 plot_surface=True):
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
        fair_sparsity: (bool) whether to make joined matrices sparse fairly
        plot_surface: (bool) Default: False
        """
        self.matrix_obj = matrix_obj
        self._n_components_create = n_components_create
        self._n_components_report = n_components_report #TODO: determine this programmatically based on scree plot
        self._approach = approach
        self._kernel = kernel
        self.n_parcels = self.matrix_obj.matrix.shape[0]
        self.n_matrices = self.matrix_obj.matrix.shape[1] // self.n_parcels
        self._sparsity = sparsity
        self._fair_sparsity = fair_sparsity
        self.columns = [f'{self.matrix_obj.short_label} G{num}' \
                        for num in range(1, self._n_components_create+1)]
        self.label = f'{self.matrix_obj.label} gradients'
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
        if (self.n_matrices > 1) & self._fair_sparsity:
            sparsity = None
            matrix = self.matrix_obj._make_sparse_fairly(sparsity=self._sparsity)
        else:
            sparsity = self._sparsity
            matrix = self.matrix_obj.matrix.values
        self.gm.fit(matrix, sparsity=sparsity)
        self.lambdas = self.gm.lambdas_
        #> add parcel labels
        self.labeled_gradients = pd.DataFrame(
            self.gm.gradients_,
            index=self.matrix_obj.matrix.index
            )
        #> project to surface
        self.surf_data = helpers.deparcellate(
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
            + ('fair' if (self._fair_sparsity & (self.n_matrices>1)) else '')\
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
            surface=self.surf_data
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
        self.surf_data = np.load(os.path.join(self.dir_path, 'gradients_surface.npz'))['surface']
        self.lambdas = np.loadtxt(os.path.join(self.dir_path, 'lambdas.txt'))

    def plot_surface(self, layout='grid', inflate=True):
        """
        Plots the gradients on the surface
        """
        for gradient_num in range(1, self._n_components_report+1):
            helpers.plot_on_bigbrain_nl(
                self.surf_data[:, gradient_num-1],
                filename=os.path.join(self.dir_path, f'surface_{layout}_G{gradient_num}.png'),
                layout=layout,
                inflate=inflate,
                plot_downsampled=True,
                #TODO: use different colors for each gradient
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
                    cmap=self.matrix_obj.cmap,
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
                    cmap=self.matrix_obj.cmap,
                    )

class MicrostructuralCovarianceGradients(Gradients):
    """
    Creates and plots gradients of microstructural covariance
    """
    def __init__(self, *args, **kwargs):
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
        fair_sparsity: (bool) whether to make joined matrices sparse fairly
        plot_surface: (bool) Default: False
        """
        super().__init__(*args, **kwargs)
        if not os.path.exists(os.path.join(self.dir_path, 'gradients_surface.npz')):
            self.plot_binned_profile()

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
            binned_parcels_laminar_thickness['bin'] = pd.cut(self.labeled_gradients[gradient_num-1], n_bins)
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

class DiseaseCovarianceGradients(Gradients):
    """
    Disease covariance gradients

    Credit: Based on Hettwer 2022 medRxiv
    """
    def perform_dominance_analysis(self, spin_permutation=False, n_perm=1000):
        """
        Performs dominance analysis to compare the contribution
        of LTC G1, MPC G1 and C.Types in explaining the variance
        of Disease G1, while testing the significance of contributions
        using spin permutation if indicated
        See https://github.com/dominance-analysis/dominance-analysis
        and Hansen 2021 bioRxiv

        Parameters
        ----------
        spin_permutation: (bool)
        n_perm: (int)
        
        Returns
        -------
        dominance_stats: (pd.DataFrame)
        dominance_pvals: (pd.DataFrame)
            if `spin_permutation` is True
        """
        #> get the gradients and downsample them
        ltcg = MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix(
                'thickness', 
                parcellation_name='aparc',
                exc_adys=self.matrix_obj.exc_adys
                )
            )
        ltcg1_surf = helpers.downsample(ltcg.surf_data)[:, :1]
        mpcg = MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix(
                'density', 
                parcellation_name='aparc',
                exc_adys=self.matrix_obj.exc_adys
                )
            )
        mpcg1_surf = helpers.downsample(mpcg.surf_data)[:, :1]
        #> get the downsampled ctypes as float
        ctypes_surf = CorticalTypes(exc_adys=self.matrix_obj.exc_adys, downsampled=True).surf_data
        #> create dataframes
        predictors = pd.DataFrame(
            np.hstack([ltcg1_surf, mpcg1_surf, ctypes_surf]),
            columns = ['LTC G1', 'MPC G1', 'Cortical Types']
        )
        outcome = pd.Series(helpers.downsample(self.surf_data[:, :1])[:, 0], name='Dis G1')
        data = pd.concat([predictors, outcome], axis=1).dropna()
        #> dominance analysis on non-permutated data
        test_dominance_analysis = Dominance(data,target='Dis G1')
        test_dominance_analysis.incremental_rsquare()
        dominance_stats = test_dominance_analysis.dominance_stats()
        if spin_permutation:
            #> create downsampled bigbrain spin permutations
            assert n_perm <= 1000
            helpers.create_bigbrain_spin_permutations(n_perm=n_perm, is_downsampled=True)
            print(f"Comparing surface data across {self.label} with spin test ({n_perm} permutations)")
            #> load the spin permutation indices
            spin_indices = np.load(os.path.join(
                SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz'
                )) # n_perm * n_vert arrays for 'lh' and 'rh'
            #> split the disease G1 in two hemispheres
            outcome_split = {
                'L': outcome.values[:outcome.shape[0]//2],
                'R': outcome.values[outcome.shape[0]//2:]
            }
            #> create the lh and rh surrogates and concatenate them
            perm_outcomes = np.concatenate([
                outcome_split['L'][spin_indices['lh']], 
                outcome_split['R'][spin_indices['rh']]
                ], axis=1) # n_perm * n_vert * n_features
            perm_dominance_stats = np.zeros((n_perm, *dominance_stats.shape))
            for perm in range(n_perm):
                print(perm)
                #> do the dominance analysis in each spin
                perm_outcome = pd.Series(perm_outcomes[perm, :], name='Dis G1')
                perm_data = pd.concat([predictors, perm_outcome], axis=1).dropna()
                perm_dominance_analysis = Dominance(data=perm_data, target='Dis G1')
                perm_dominance_analysis.incremental_rsquare()
                perm_dominance_stats[perm, :, :] = perm_dominance_analysis.dominance_stats().values
            #> save null distribution for future reference
            np.savez_compressed(
                os.path.join(self.dir_path, 'dominance_null.npz'),
                perm_dominance_stats
            )
            #> calculate one-sided p-vals (all values are positive so no difference between one or two-sided)
            dominance_pvals = (perm_dominance_stats > dominance_stats.values[np.newaxis, :, :]).mean(axis=0)
            dominance_pvals = pd.DataFrame(
                dominance_pvals,
                columns=dominance_stats.columns,
                index=dominance_stats.index
                )
            #> calculate p-value for the total variance explained
            # by the model (which is the sum of 'Total Dominance' 
            # for all variables)
            perm_total_var = perm_dominance_stats[:, :, 3].sum(axis=1)
            dominance_stats.loc['Sum', 'Total Dominance'] \
                = dominance_stats.loc[:, 'Total Dominance'].sum()
            dominance_pvals.loc['Sum', 'Total Dominance'] \
                = (perm_total_var > dominance_stats.loc['Sum', 'Total Dominance']).mean()
            return dominance_stats, dominance_pvals
        else:
            dominance_stats.loc['Sum', 'Total Dominance'] \
                = dominance_stats.loc[:, 'Total Dominance'].sum()
            return dominance_stats

class StructuralFeatures(ContCorticalSurface):
    label = 'Structural features'
    def __init__(self, correct_curvature):
        """
        Structural features including total/laminar thickness, laminar densities and
        microstructural profile moments
        
        Parameters
        ----------
        correct_curvature: (str or None)
            - 'volume' [default]: normalize relative thickness by curvature according to the 
            equivolumetric principle i.e., use relative laminar volume instead of relative laminar 
            thickness. Laminar volume is expected to be less affected by curvature.
            - 'regress': regresses map of curvature out from the map of relative thickness
            of each layer (separately) using a linear regression. This is a more lenient approach of 
            controlling for curvature.
            - None
        """
        self.correct_curvature = correct_curvature
        self.dir_path = OUTPUT_DIR
        self._create()
    
    def _create(self):
        self.columns = []
        features = []
        #> Total cortical thickness
        abs_laminar_thickness = datasets.load_laminar_thickness(
            regress_out_curvature=False,
            normalize_by_total_thickness=False
        )
        abs_laminar_thickness = np.concatenate(
            [abs_laminar_thickness['L'], abs_laminar_thickness['R']], axis=0
            )
        total_thickness = abs_laminar_thickness.sum(axis=1)[:, np.newaxis]
        features.append(total_thickness)
        self.columns += ['Total thickness']
        #> Relative thicknesses/volumes
        if self.correct_curvature == 'volume':
            rel_laminar_thickness = datasets.load_laminar_volume()
        elif self.correct_curvature == 'regress':
            rel_laminar_thickness = datasets.load_laminar_thickness(
                regress_out_curvature=True,
                normalize_by_total_thickness=True,
                )
        else:
            rel_laminar_thickness = datasets.load_laminar_thickness(
                regress_out_curvature=False,
                normalize_by_total_thickness=True,
                )
        rel_laminar_thickness = np.concatenate(
            [rel_laminar_thickness['L'], rel_laminar_thickness['R']], axis=0
            )
        features.append(rel_laminar_thickness)
        self.columns += [f'Layer {num} relative thickness' for num in range(1, 7)]
        #> Deep thickness ratio
        deep_thickness_ratio = rel_laminar_thickness[:, 3:].sum(axis=1)[:, np.newaxis]
        features.append(deep_thickness_ratio)
        self.columns += ['Deep laminar thickness ratio']
        #> Laminar densities
        laminar_density = datasets.load_laminar_density()
        laminar_density = np.concatenate(
            [laminar_density['L'], laminar_density['R']], axis=0
            )
        features.append(laminar_density)
        self.columns += [f'Layer {num} density' for num in range(1, 7)]
        #> Microstructural profile moments
        density_profiles = datasets.load_total_depth_density()
        density_profiles = np.concatenate(
            [density_profiles['L'], density_profiles['R']], axis=0
            )
        features += [
            np.mean(density_profiles, axis=1)[:, np.newaxis],
            np.std(density_profiles, axis=1)[:, np.newaxis],
            scipy.stats.skew(density_profiles, axis=1)[:, np.newaxis],
            scipy.stats.kurtosis(density_profiles, axis=1)[:, np.newaxis]
        ]
        self.columns += ['Density mean', 'Density std', 'Density skewness', 'Density kurtosis']
        #> concatenate all the features into a single array
        self.surf_data = np.hstack(features)

class DiseaseMaps(ContCorticalSurface):
    """
    Maps of cortical thickness differences in disorders from ENIGMA 
    mega-/meta-analyses. When different options are provided, adult
    population, mega-analyses, and combined disorder subtypes are
    loaded.
    """
    def __init__(self):
        """
        Loads disorder maps from ENIGMA toolbox and projects
        them back to surface
        """
        #> load disorder maps from ENIGMA toolbox
        # (in DK parcellation)
        self.parcellated_disorder_maps = datasets.load_disease_maps()
        #> project it back to surface
        self.surf_data = helpers.deparcellate(self.parcellated_disorder_maps, 'aparc')
        self.columns = self.parcellated_disorder_maps.columns

class CatCorticalSurface:
    """
    Generic class for cortical surfaces with categorical data
    (cortical types and Yeo networks)
    """
    def _parcellate_and_categorize(self, surf_data, columns, parcellation_name):
        """
        Parcellates other.surf_data and adds a column for the category of
        each parcel

        Parameters
        ---------
        surf_data: (np.ndarray) 
            n_vert x n_features array of continous surface data
        columns: (list of str) 
            names of columns corresponding to surf_data
        parcellation_name: (str)

        Returns
        --------
        parcellated_data (pd.DataFrame)
        """
        #> parcellate the continous data
        parcellated_data = helpers.parcellate(surf_data, parcellation_name)
        parcellated_data.columns = columns
        #> add the category of parcels to the parcellated data
        is_downsampled = (surf_data.shape[0] == datasets.N_VERTICES_HEM_BB_ICO5*2)
        parcellated_data.loc[:, self.label] = self.load_parcels_categories(parcellation_name, downsampled=is_downsampled)
        #> exclude unwanted categories
        parcellated_data = parcellated_data[~parcellated_data[self.label].isin(self.excluded_categories)]
        parcellated_data[self.label] = parcellated_data[self.label].cat.remove_unused_categories()
        #> remove NaNs (usually from surf_data since NaN is an unwanted category)
        parcellated_data = parcellated_data.dropna()
        return parcellated_data

    def _plot_raincloud(self, parcellated_data, column, out_dir):
        """
        Plots the raincloud of the `column` from `parcellated_data`
        across `self.included_categories` and saves it in `out_dir`
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = ptitprince.RainCloud(
            data=parcellated_data, 
            y=column,
            x=self.label,
            palette=self.colors,
            bw = 0.2, width_viol = 1, 
            orient = 'h', move = 0.2, alpha = 0.4,
            ax=ax)
        sns.despine(ax=ax, offset=10, trim=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{column}_raincloud'), dpi=192)
        #> colorbar with the correct vmin and vmax
        clbar_fig = helpers.make_colorbar(
            ax.get_xticks()[0], ax.get_xticks()[-1], 
            figsize=(4, 4), orientation='horizontal')
        clbar_fig.tight_layout()
        clbar_fig.savefig(os.path.join(out_dir, f'{column}_raincloud_clbar'), dpi=192)

    def _plot_binned_stacked_bar(self, parcellated_data, column, out_dir, nbins):
        """
        Plots the binned stacked of the `column` from `parcellated_data`
        across `self.included_categories` and saves it in `out_dir`
        """
        #>> assign each vertex to one of the 10 bins
        _, bin_edges = np.histogram(parcellated_data.loc[:,column], bins=nbins)
        parcellated_data[f'{column}_bin'] = np.digitize(parcellated_data.loc[:,column], bin_edges[:-1])
        #>> calculate ratio of categories in each bin
        bins_categories_counts = (parcellated_data
                            .groupby([f'{column}_bin',self.label])
                            .size().unstack(fill_value=0))
        bins_categories_freq = bins_categories_counts.divide(bins_categories_counts.sum(axis=1), axis=0)
        #>> plot stacked bars at each bin showing freq of the cortical categories
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x = bins_categories_freq.index,
            height = bins_categories_freq.iloc[:, 0],
            width = 0.95,
            color=self.colors[0],
            label=self.included_categories[0]
            )
        for type_idx in range(1, self.n_categories):
            ax.bar(
                x = bins_categories_freq.index,
                height = bins_categories_freq.iloc[:, type_idx],
                width = 0.95,
                bottom = bins_categories_freq.cumsum(axis=1).iloc[:, type_idx-1],
                color=self.colors[type_idx],
                label=self.included_categories[type_idx]
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'{column} bins')
        ax.set_ylabel(f'Proportion of {self.label.lower()}')
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{column}_stacked_bar'), dpi=192)
        clfig = helpers.make_colorbar(
            parcellated_data[column].min(), parcellated_data[column].max(),
            bins=nbins, orientation='horizontal', figsize=(6,4))
        clfig.savefig(os.path.join(out_dir, f'{column}_stacked_bar_clbar'), dpi=192)

    def _anova(self, parcellated_data, column, output='text', force_posthocs=False):
        """
        Compares value of parcellated data across `self.included_categories` 
        using ANOVA and post-hoc t-tests

        Parameters
        ---------
        parcellated_data: (pd.DataFrame) 
            with parcellated surface data and categories
        column: (str) 
            the selected column
        output: (str)
            - text: returns human readable text of the results
            - stats: returns F and T stats
        force_posthocs: (bool)
            do and report post-hocs regardless of p-values

        Returns
        ---------
        anova_res_str: (str) 
            results of anova and post-hocs as text
        OR
        anova_res: (pd.Series) 
            F and T stats of anova and post-hocs
        """
        F, p_val = (scipy.stats.f_oneway(*[
                                    category_data[1][column].dropna().values \
                                    for category_data in parcellated_data.groupby(self.label)
                                    ]))
        anova_res_str = f'----\n{column}: F statistic {F}, pvalue {p_val}\n'
        anova_res = pd.Series({'F': F})
        if (p_val < 0.05) or force_posthocs:
            # set alpha for bonferroni correction across all pairs of categories
            alpha = 0.05 / len(list(itertools.combinations(parcellated_data[self.label].cat.categories, 2)))
            if force_posthocs:
                anova_res_str += f"\tPost-hoc T-tests:\n(bonferroni alpha: {alpha})\n"
            else:
                anova_res_str += f"\tPost-hoc T-tests passing alpha of {alpha}:\n"
            for cat1, cat2 in itertools.combinations(parcellated_data[self.label].cat.categories, 2):
                t_statistic, t_p = scipy.stats.ttest_ind(
                    parcellated_data.loc[parcellated_data[self.label]==cat1, column].dropna(),
                    parcellated_data.loc[parcellated_data[self.label]==cat2, column].dropna(),
                )
                anova_res.loc[f'{cat1}-{cat2}'] = t_statistic
                if (t_p < alpha) or (force_posthocs):
                    anova_res_str += f"\t\t{cat1} - {cat2}: T {t_statistic}, p {t_p}\n"
        if output == 'text':
            return anova_res_str
        else:
            return anova_res

    def compare(self, other, parcellation_name, nbins=10):
        """
        Compares the difference of `other.surf_data` across `self.included_categories` and 
        plots it as raincloud or stacked bar plots

        Parameters
        ----------
        other: (ContCorticalSurface)
        parcellation_name: (str)
        exc_adys: (bool)
            exclude allocortical and a-/dysgranular regions
        nbins: (int)
            number of bins in the binned stacked bar plot
        """
        #> parcellate data and specify cortical type of each parcel
        parcellated_data = self._parcellate_and_categorize(other.surf_data, other.columns, parcellation_name)
        #> specify output dir
        out_dir = os.path.join(other.dir_path, f'association_{self.label.lower().replace(" ", "_")}')
        os.makedirs(out_dir, exist_ok=True)
        #> investigate the association of gradient values and cortical categories (visually and statistically)
        anova_res_str = "ANOVA Results\n--------\n"
        for column in other.columns:
            # 1) Raincloud plot
            self._plot_raincloud(parcellated_data, column, out_dir)
            #> 2) Binned stacked bar plot
            self._plot_binned_stacked_bar(parcellated_data, column, out_dir, nbins)
            #> ANOVA
            anova_res_str += self._anova(parcellated_data, column, output='text')
        print(anova_res_str)
        with open(os.path.join(out_dir, 'anova.txt'), 'w') as anova_res_file:
            anova_res_file.write(anova_res_str)
    
    def spin_compare(self, other, parcellation_name, n_perm=1000):
        """
        Compares the difference of `other.surf_data` across `self.included_categories` 
        while correcting for spatial autocorrelation

        Note: This is very computationally intensive and slow.

        Parameters
        ----------
        other: (ContCorticalSurface)
        parcellation_name: (str)
        n_perm: (int)

        Returns
        ---------
        spin_pvals: (pd.DataFrame) n_stats (e.g. F, EU3-EU2, ...) x n_columns
        """
        #> create downsampled bigbrain spin permutations
        assert n_perm <= 1000
        helpers.create_bigbrain_spin_permutations(n_perm=n_perm, is_downsampled=True)
        print(f"Comparing surface data across {self.label} with spin test ({n_perm} permutations)")
        #> split the other surface in two hemispheres
        surf_data = {
            'L': other.surf_data[:other.surf_data.shape[0]//2],
            'R': other.surf_data[other.surf_data.shape[0]//2:]
        }
        #> load the spin permutation indices
        spin_indices = np.load(os.path.join(
            SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz'
            )) # n_perm * n_vert arrays for 'lh' and 'rh'
        #> create the lh and rh surrogates and concatenate them
        surrogates = np.concatenate([
            surf_data['L'][spin_indices['lh']], 
            surf_data['R'][spin_indices['rh']]
            ], axis=1) # n_perm * n_vert * n_features
        #> add the original surf_data at the beginning
        downsampled_surf_data_and_surrogates = np.concatenate([
            helpers.downsample(other.surf_data)[np.newaxis, :, :],
            surrogates
            ], axis = 0)
        all_anova_results = {column:[] for column in other.columns}
        for perm in range(n_perm+1):
            print(perm)
            #> get the F stat and T stats for each permutation
            curr_data = downsampled_surf_data_and_surrogates[perm, :, :]
            curr_parcelated_data = self._parcellate_and_categorize(curr_data, other.columns, parcellation_name)
            for column in other.columns:
                all_anova_results[column].append(
                    self._anova(curr_parcelated_data, column, output='stats', force_posthocs=True)
                )
        spin_pvals = pd.DataFrame()
        for column in other.columns:
            #> calculate the two-sided non-parametric p-values as the
            #  ratio of permutated F stat and T stats more extreme
            #  to the non-permutated values (in index -1)
            curr_column_res = pd.concat(all_anova_results[column], axis=1).T # n_perm x n_stats
            spin_pvals.loc[:, column] = (np.abs(curr_column_res.iloc[:-1, :]) >= np.abs(curr_column_res.iloc[-1, :])).mean(axis=0)
        return spin_pvals


class CorticalTypes(CatCorticalSurface):
    """
    Map of cortical types
    """
    label = 'Cortical Type'
    def __init__(self, exc_adys=True, downsampled=False):
        """
        Loads the map of cortical types
        """
        self.exc_adys = exc_adys
        self.colors = sns.color_palette("RdYlGn_r", 6)
        if self.exc_adys:
            self.included_categories = ['EU1', 'EU2', 'EU3', 'KO']
            self.excluded_categories = [np.NaN, 'ALO', 'AG', 'DG']
            self.colors = self.colors[2:]
        else:
            self.included_categories = ['AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO']
            self.excluded_categories = [np.NaN, 'ALO']
        self.n_categories = len(self.included_categories)
        # load unparcellated surface data
        cortical_types_map = datasets.load_cortical_types(downsampled=downsampled)
        self.surf_data = cortical_types_map.cat.codes.values.reshape(-1, 1).astype('float')
        self.surf_data[cortical_types_map.isin(self.excluded_categories), 0] = np.NaN
        self.columns = ['Cortical Type']

    def load_parcels_categories(self, parcellation_name, downsampled):
        return datasets.load_cortical_types(parcellation_name, downsampled=downsampled)

    
class YeoNetworks(CatCorticalSurface):
    """
    Map of Yeo networks
    """
    label = 'Resting state network'
    def __init__(self):
        """
        Loads the map of Yeo networks
        """
        self.colors = self._load_colormap()
        self.excluded_categories = [np.NaN, 'None']
        self.included_categories = [
            'Visual', 'Somatomotor', 'Dorsal attention', 
            'Ventral attention', 'Limbic', 'Frontoparietal', 'Default'
            ]
        self.n_categories = len(self.included_categories)
        # load unparcellated surface data
        yeo_map = datasets.load_yeo_map()
        self.surf_data = yeo_map.cat.codes.values.reshape(-1, 1).astype('float')
        self.surf_data[yeo_map.isin(self.excluded_categories), 0] = np.NaN
        self.columns = ['Resting state network']

    def load_parcels_categories(self, parcellation_name, downsampled):
        return datasets.load_yeo_map(parcellation_name, downsampled=downsampled)

    def _load_colormap(self):
        """
        Load the colormap of 7 Yeo networks
        """
        yeo_giftii = nibabel.load(
            os.path.join(
                SRC_DIR,
                'tpl-bigbrain_hemi-L_desc-Yeo2011_7Networks_N1000.label.gii')
            )
        yeo_colors = [l.rgba[:-1] for l in yeo_giftii.labeltable.labels[1:]]
        return sns.color_palette(yeo_colors, as_cmap=True)