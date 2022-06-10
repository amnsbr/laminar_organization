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
import nilearn.surface
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from dominance_analysis import Dominance
import subprocess
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import helpers
import datasets
import matrices

# specify directories
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
WB_PATH = os.path.join(cwd, '..', 'tools', 'workbench', 'bin_linux64', 'wb_command')


class ContCorticalSurface:
    """
    Generic class for common functions on cortical surfaces with continuous data
    """
    cmap = 'viridis' # default
    # TODO: specify required fields and methods
    def __init__(self, surf_data, columns, label, dir_path=None, 
                 cmap='viridis', parcellation_name=None):
        """
        Initialize object using n-d surf_data. This will be overwritten
        by the sub-classes

        Parameters
        ----------
        surf_data: (np.ndarray) n_vert [ico5 or ico7] x n_features
        columns: (list of str)
        label: (str)
        dir_path: (str)
        """
        self.surf_data = surf_data
        # downsample it
        if surf_data.shape[0] == datasets.N_VERTICES_HEM_BB * 2:
            self.surf_data = helpers.downsample(self.surf_data)
        # make it 2d
        if self.surf_data.ndim == 1:
            self.surf_data = self.surf_data[:, np.newaxis]
        self.columns = columns
        self.label = label
        self.dir_path = dir_path
        if self.dir_path is None:
            self.dir_path = OUTPUT_DIR
        os.makedirs(self.dir_path, exist_ok=True)
        self.cmap = cmap
        self.parcellation_name = parcellation_name
        if self.parcellation_name:
            self.parcellated_data = helpers.parcellate(
                self.surf_data, self.parcellation_name
                )
            self.parcellated_data.columns = self.columns


    def plot(self, columns=None, add_labels=None, cmap=None, **plotter_kwargs):
        """
        Plot the map on surface

        Parameters
        ---------
        columns: (list | None)
        add_labels: (str | None)
            - None
            - 'top'
            - 'bottom'
            - 'left'
            - 'right'
        cmap: (None | str) if None uses the default

        Returns
        -------
        plots: (list of figure objects or the path to them)
        """
        if columns is None:
            columns = self.columns
        data = pd.DataFrame(self.surf_data, columns=self.columns)
        plots = []
        for column in columns:
            if add_labels:
                label_text = {add_labels:[column]}
            else:
                label_text = None
            if cmap is None:
                cmap = self.cmap
            plot = helpers.plot_surface(
                data.loc[:, column].values,
                os.path.join(self.dir_path, f'surface_{column}'),
                label_text = label_text,
                cmap = cmap,
                **plotter_kwargs    
            )
            plots.append(plot)
        return plots

    def correlate(self, other, parcellated=True, n_perm=1000,
                 x_columns=None, y_columns=None, axis_off=False,
                 sort_barplot=True):
        """
        Calculate the correlation of surface maps with permutation test
        (with surrogates created using spins or variograms) and plot 
        the scatter plots

        Parameters
        ---------
        self, other: (CorticalSurface) with surf_data fields which are n_vert x n_features np.ndarrays
        parcellated: (bool)
            - True: correlate across parcels (uses variograms for permutation)
            - False: correlate across vertices (uses spin test for permutation)
        n_perm: (int) number of permutations. Max: 1000
        x_columns, y_columns: (list of str) selected columns from self (x) and other (y)
        axis_off: (bool) turn the axis off from the plot
        sort_barplot: (bool) sort bar plot based on correlation values
        """
        # TODO: add an option for removing outliers
        if parcellated:
            assert self.parcellation_name == other.parcellation_name
            assert self.parcellation_name is not None
        out_dir = os.path.join(
            self.dir_path, 
            'correlation_'\
            +('parcellated_' if parcellated else '')\
            + other.label.lower().replace(" ", "_"))
        # TODO: this naming approach sometimes can lead
        # to error because of the file name being too long
        os.makedirs(out_dir, exist_ok=True)
        # select columns
        if not x_columns:
            x_columns = self.columns
        if not y_columns:
            y_columns = other.columns
        # use parcellated or downsampled data of selected columns
        if parcellated:
            shared_parcels = self.parcellated_data.index.intersection(other.parcellated_data.index)
            x_data = self.parcellated_data.loc[shared_parcels, x_columns]
            y_data = other.parcellated_data.loc[shared_parcels, y_columns]
        else:
            x_data = pd.DataFrame(
                self.surf_data, 
                columns=self.columns
                ).loc[:, x_columns]
            y_data = pd.DataFrame(
                other.surf_data, 
                columns=other.columns
                ).loc[:, y_columns]
        if parcellated:
            # statistical test using variogram-based permutation
            print("Calculating correlations using variogram-based permutation")
            coefs, pvals, coefs_null_dist =  helpers.variogram_test(
                X = x_data, 
                Y = y_data,
                parcellation_name = self.parcellation_name,
                n_perm = n_perm,
                surrogates_path = os.path.join(self.dir_path, f'variogram_surrogates_{"-".join(x_columns)}')
                )
        else:
            # statistical test using spin permutation
            print("Calculating correlations with spin test")
            coefs, pvals, coefs_null_dist =  helpers.spin_test(
                surface_data_to_spin = x_data.values, 
                surface_data_target = y_data.values,
                n_perm=n_perm,
                is_downsampled=True
                )
        # save null distribution for future reference
        np.savez_compressed(
            os.path.join(out_dir, 'coefs_null_dist.npz'),
            coefs_null_dist=coefs_null_dist
        )
        # clean and save the results
        coefs = pd.DataFrame(coefs)
        pvals = pd.DataFrame(pvals)
        coefs.columns = pvals.columns = x_columns
        coefs.index = pvals.index = y_columns
        coefs.to_csv(os.path.join(out_dir, 'coefs.csv'))
        pvals.to_csv(os.path.join(out_dir, 'pvals.csv'))
        # bar plots
        for x_column in x_columns:
            curr_coefs = coefs.loc[:, x_column]
            curr_pvals = pvals.loc[:, x_column]
            if sort_barplot:
                curr_coefs = curr_coefs.sort_values()
                curr_pvals = curr_pvals.loc[curr_coefs.index]
            # set the bar color based on p-value
            # white = non-sig; darker = lower p-val
            colors=[(1, 1, 1, 0)]*len(y_columns)
            for i, y_column in enumerate(curr_pvals.index):
                if curr_pvals[y_column] < 0.05:
                    colors[i] = tuple([curr_pvals[y_column]/0.05]*3 + [0])
            sns.set_style("white")
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.barplot(
                data=curr_coefs.to_frame(), 
                x=x_column, y=curr_coefs.index, 
                palette=colors, edgecolor=".2", ax=ax)
            ax.axvline(0, color='black')
            sns.despine(offset=1, left=True, trim=False, ax=ax)
            ax.set_xlim((-1, 1))
            ax.set_ylabel('')
            ax.set_xlabel(f'Correlation with {x_column}')
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    out_dir, 
                    f'barplot_{x_column.lower().replace(" ", "_")}.png'),
                dpi=192
                )
        # regression plots
        for x_column in x_columns:
            for y_column in y_columns:
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.regplot(
                    x=x_data.loc[:,x_column], 
                    y=y_data.loc[:,y_column],
                    scatter_kws=dict(alpha=0.2, s=5, color='grey'),
                    line_kws=dict(color='red'),
                    ax=ax)
                sns.despine(offset=10, trim=True, ax=ax)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                # add correlation coefficients and p vals on the figure
                text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
                text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
                ax.text(text_x, text_y, 
                        f'r = {coefs.loc[y_column, x_column]:.2f}; $\mathregular{{p_{{spin}}}}$ = {pvals.loc[y_column, x_column]:.2f}',
                        color='black',
                        size=14,
                        multialignment='left')
                if axis_off:
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        out_dir, 
                        f'{x_column.lower().replace(" ", "_")}-'\
                        + f'{y_column.lower().replace(" ", "_")}.png'),
                    dpi=192
                    )
    
    def microstructure_spatial_cross_validation(self, col_idx=0, exc_regions='adysgranular'):
        """
        Caclulates R2 of predicting surf_data[:, col_idx] 
        based on LTC G1, MPC G1 and C.Types after doing spatial
        leave one out cross-validation

        Parameters
        --------
        exc_regions: (str | None) which regions to exclude from structural maps
        
        Ref: https://arxiv.org/abs/2005.14263
        """
        # Make sure it's only called with surfaces other than
        # microstructure gradients
        assert not isinstance(self, MicrostructuralCovarianceGradients)
        # Load parcellated predictors and outcome
        ltcg1 = MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix(
                'thickness', 
                parcellation_name=self.parcellation_name,
                exc_regions=exc_regions
                )
            ).parcellated_data.iloc[:, 0].rename('LTC G1')
        mpcg1 = MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix(
                'density', 
                parcellation_name=self.parcellation_name,
                exc_regions=exc_regions
                )
            ).parcellated_data.iloc[:, 0].rename('MPC G1')
        ctypes = (
            datasets.load_cortical_types('aparc')
            .loc[ltcg1.index] # select valid parcels
            .cat.codes # convert to int
            .rename('Cortical Types')
        )
        Xs = {
            'Combined': pd.concat(
                [ltcg1, mpcg1, ctypes],
                axis=1,
                ),
            'LTC G1': ltcg1.to_frame(),
            'MPC G1': mpcg1.to_frame(),
            'Cortical Types': ctypes.to_frame()
        }
        y = (
            self.parcellated_data
            .iloc[:, col_idx] # select the column
            .loc[ltcg1.index] # select the same parcels as predictors
            .rename('DV').to_frame()
        )
        # get the neighbors of each parcel so that
        # they can be removed from training dataset
        # in each iteration
        neighbors = helpers.get_neighbors_mask(
            'aparc', 
            proportion=0.2,
            exc_regions=exc_regions,
        )
        neighbors = neighbors.loc[neighbors.index, neighbors.index]
        res = pd.DataFrame()
        for model_name, X in Xs.items():
            # get predicted y based on spatial leave-one-out CV
            y_pred = pd.Series(index=y.index)
            for parcel, parcel_neighbors in neighbors.iterrows():
                #> at each iteration select the seed
                # parcel as the test data
                X_test = X.loc[[parcel], :]
                #> and select the rest of the brain
                # except seed and its neighbors as train
                X_train = X.loc[~parcel_neighbors, :]
                y_train = y.loc[~parcel_neighbors, :]
                #> run linear regression and predict y at
                # seed
                lr = LinearRegression().fit(X_train, y_train)
                y_pred.loc[parcel] = lr.predict(X_test).flatten()[0]
            # calculate r2 and corr
            res.loc[model_name, 'r2'] = r2_score(y.values.flatten(), y_pred.values.flatten())
            res.loc[model_name, 'corr(pred,true)'] = np.corrcoef(y.values.flatten(), y_pred.values.flatten())[0, 1]
        res.to_csv(
            os.path.join(
                self.dir_path, 
                f'microstructure_cross_validation_res_col-{self.columns[col_idx]}.csv'
                )
            )

    def microstructure_dominance_analysis(self, col_idx=0, n_perm=1000, exc_regions='adysgranular'):
        """
        Performs dominance analysis to compare the contribution
        of LTC G1, MPC G1 and C.Types in explaining the variance
        of surf_data[:, col_idx], while testing the significance of contributions
        using spin permutation if indicated
        See https://github.com/dominance-analysis/dominance-analysis
        and Hansen 2021 bioRxiv

        Parameters
        ----------
        col_idx: (int) column index to use as the dependent variable
        n_perm: (int) if zero spin permutation is not  performed
        exc_regions: (str | None) which regions to exclude from structural maps
        """
        # TODO: add the option for doing this on parcellated data
        # and using variogram permutation
        # make sure it's only called with surfaces other than
        # microstructure gradients
        assert not isinstance(self, MicrostructuralCovarianceGradients)
        # specify output path
        out_path = os.path.join(
            self.dir_path, 
            'microstructure_dominance_analysis' \
            + f'_col-{self.columns[col_idx]}' \
            + (f'_exc-{exc_regions}' if exc_regions else '') \
            + f'_nperm-{n_perm}'
            )
        os.makedirs(out_path, exist_ok=True)
        # get and downsample the disease gradient
        # Note: it is important not to exclude any regions from disease gradient,
        # as it will get spun and disease g NaNs at the excluded regions move around
        # but not the NaNs at the excluded regions in the microstructural data, and
        # this can create up to two-fold points where either disease or microstructure
        # is NaN
        outcome = pd.Series(self.surf_data[:, 0], name='DV')
        # get the structural gradients and downsample them
        ltcg = MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix(
                'thickness', 
                parcellation_name=self.parcellation_name,
                exc_regions=exc_regions
                )
            )
        ltcg1_surf = ltcg.surf_data[:, :1]
        mpcg = MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix(
                'density', 
                parcellation_name=self.parcellation_name,
                exc_regions=exc_regions
                )
            )
        mpcg1_surf = mpcg.surf_data[:, :1]
        # get the downsampled ctypes as float
        ctypes_surf = CorticalTypes(exc_regions=exc_regions, downsampled=True).surf_data
        # create dataframes
        predictors = pd.DataFrame(
            np.hstack([ltcg1_surf, mpcg1_surf, ctypes_surf]),
            columns = ['LTC G1', 'MPC G1', 'Cortical Types']
        )
        data = pd.concat([predictors, outcome], axis=1).dropna()
        # dominance analysis on non-permutated data
        test_dominance_analysis = Dominance(data,target='DV')
        test_dominance_analysis.incremental_rsquare()
        dominance_stats = test_dominance_analysis.dominance_stats()
        if n_perm > 0:
            # create downsampled bigbrain spin permutations
            assert n_perm <= 1000
            helpers.create_bigbrain_spin_permutations(n_perm=n_perm, is_downsampled=True)
            # load the spin permutation indices
            spin_indices = np.load(os.path.join(
                SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz'
                )) # n_perm * n_vert arrays for 'lh' and 'rh'
            # split the disease G1 in two hemispheres
            outcome_split = {
                'L': outcome.values[:outcome.shape[0]//2],
                'R': outcome.values[outcome.shape[0]//2:]
            }
            # create the lh and rh surrogates and concatenate them
            perm_outcomes = np.concatenate([
                outcome_split['L'][spin_indices['lh']], 
                outcome_split['R'][spin_indices['rh']]
                ], axis=1) # n_perm * n_vert * n_features
            perm_dominance_stats = np.zeros((n_perm, *dominance_stats.shape))
            for perm in range(n_perm):
                logging.info(perm)
                # do the dominance analysis in each spin
                perm_outcome = pd.Series(perm_outcomes[perm, :], name='DV')
                perm_data = pd.concat([predictors, perm_outcome], axis=1).dropna()
                perm_dominance_analysis = Dominance(data=perm_data, target='DV')
                perm_dominance_analysis.incremental_rsquare()
                perm_dominance_stats[perm, :, :] = perm_dominance_analysis.dominance_stats().values
            # save null distribution for future reference
            np.savez_compressed(
                os.path.join(out_path, 'null.npz'),
                perm_dominance_stats
            )
            # calculate one-sided p-vals (all values are positive so no difference between one or two-sided)
            dominance_pvals = (perm_dominance_stats > dominance_stats.values[np.newaxis, :, :]).mean(axis=0)
            dominance_pvals = pd.DataFrame(
                dominance_pvals,
                columns=dominance_stats.columns,
                index=dominance_stats.index
                )
            # calculate p-value for the total variance explained
            # by the model (which is the sum of 'Total Dominance' 
            # for all variables)
            perm_total_var = perm_dominance_stats[:, :, 3].sum(axis=1)
            dominance_stats.loc['Sum', 'Total Dominance'] \
                = dominance_stats.loc[:, 'Total Dominance'].sum()
            dominance_pvals.loc['Sum', 'Total Dominance'] \
                = (perm_total_var > dominance_stats.loc['Sum', 'Total Dominance']).mean()
            dominance_pvals.to_csv(os.path.join(out_path, 'dominance_pvals.csv'))
        else:
            dominance_stats.loc['Sum', 'Total Dominance'] \
                = dominance_stats.loc[:, 'Total Dominance'].sum()
        dominance_stats.to_csv(os.path.join(out_path, 'dominance_stats.csv'))

class LaminarFeatures(ContCorticalSurface):
    label = 'Laminar features'
    def __init__(self, parcellation_name=None, correct_curvature='smooth-10', exc_regions=None):
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
        self.parcellation_name = parcellation_name
        self.correct_curvature = correct_curvature
        self.exc_regions = exc_regions
        self.dir_path = os.path.join(OUTPUT_DIR, 'laminar_features')
        os.makedirs(self.dir_path, exist_ok=True)
        self.file_path = os.path.join(
            self.dir_path, 
            f'curv-{correct_curvature}_exc-{exc_regions}')
        if os.path.exists(self.file_path+'.npz'):
            npz = np.load(self.file_path+'.npz', allow_pickle=True)
            self.columns = npz['columns']
            self.surf_data = npz['surf_data']
        else:
            self._create()
            np.savez_compressed(
                self.file_path+'.npz',
                surf_data = self.surf_data,
                columns = self.columns
            )
        if self.parcellation_name:
            self.parcellated_data = helpers.parcellate(
                self.surf_data,
                self.parcellation_name
            )
            self.parcellated_data.columns = self.columns
    
    def _create(self):
        self.columns = []
        features = []
        # Absolute laminar thickness
        abs_laminar_thickness = datasets.load_laminar_thickness(
            normalize_by_total_thickness=False,
            exc_regions=self.exc_regions
        )
        abs_laminar_thickness = helpers.downsample(
            np.concatenate(
                [abs_laminar_thickness['L'], abs_laminar_thickness['R']], axis=0
                )
        )
        features.append(abs_laminar_thickness)
        self.columns += [f'Layer {num} absolute thickness' for num in range(1, 7)]
        # Total cortical thickness
        total_thickness = abs_laminar_thickness.sum(axis=1)[:, np.newaxis]
        total_thickness = helpers.downsample(total_thickness)
        features.append(total_thickness)
        self.columns += ['Total thickness']
        #> Relative thicknesses/volumes
        if self.correct_curvature is None:
            rel_laminar_thickness = datasets.load_laminar_thickness(exc_regions=self.exc_regions)
        elif 'smooth' in self.correct_curvature:
            smooth_disc_radius = int(self.correct_curvature.split('-')[1])
            rel_laminar_thickness = datasets.load_laminar_thickness(
                exc_regions=self.exc_regions,
                smooth_disc_radius=smooth_disc_radius
            )
        elif self.correct_curvature == 'regress':
            rel_laminar_thickness = datasets.load_laminar_thickness(
                exc_regions=self.exc_regions,
                regress_out_curvature=True,
            )
        if rel_laminar_thickness['L'].shape[0] == datasets.N_VERTICES_HEM_BB:
            rel_laminar_thickness = helpers.downsample(rel_laminar_thickness)
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
        laminar_density = datasets.load_laminar_density(exc_regions=self.exc_regions)
        laminar_density = np.concatenate(
            [laminar_density['L'], laminar_density['R']], axis=0
            )
        laminar_density = helpers.downsample(laminar_density)
        features.append(laminar_density)
        self.columns += [f'Layer {num} density' for num in range(1, 7)]
        #> Microstructural profile moments
        density_profiles = datasets.load_total_depth_density(exc_regions=self.exc_regions)
        density_profiles = np.concatenate([density_profiles['L'], density_profiles['R']])
        density_profiles = helpers.downsample(density_profiles)
        features += [
            np.mean(density_profiles, axis=1)[:, np.newaxis],
            np.std(density_profiles, axis=1)[:, np.newaxis],
            scipy.stats.skew(density_profiles, axis=1)[:, np.newaxis],
            scipy.stats.kurtosis(density_profiles, axis=1)[:, np.newaxis]
        ]
        self.columns += ['Density mean', 'Density std', 'Density skewness', 'Density kurtosis']
        #> concatenate all the features into a single array
        self.surf_data = np.hstack(features)

class Gradients(ContCorticalSurface):
    """
    Generic class for creating and plotting gradients
    """
    def __init__(self, matrix_obj, n_components_create=10, n_components_report=2,
                 approach='dm', kernel='normalized_angle', sparsity=0.9, fair_sparsity=True,
                 hemi=None, cmap='viridis', create_plots=True):
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
        hemi: (str | None) limit the gradient to selected hemisphere ('L' or 'R')
        create_plots: (bool) Default: False
        """
        self.matrix_obj = matrix_obj
        self.parcellation_name = self.matrix_obj.parcellation_name
        self._n_components_create = n_components_create
        self._n_components_report = n_components_report #TODO: determine this programmatically based on scree plot
        self._approach = approach
        self._kernel = kernel
        self.n_parcels = self.matrix_obj.matrix.shape[0]
        self.n_matrices = self.matrix_obj.matrix.shape[1] // self.n_parcels
        self._sparsity = sparsity
        self._fair_sparsity = fair_sparsity
        self.hemi = hemi
        if (self.matrix_obj.split_hem) & (self.hemi is None):
            # in case the matrix is split in two hemispheres
            # and this is the parent gradient where hemispheres
            # are concatenated do not plot anything as some
            # of the fields (e.g. lambdas) are undefined
            create_plots = False
        self.columns = [f'{self.matrix_obj.short_label} G{num}' \
                        for num in range(1, self._n_components_create+1)]
        self.label = f'{self.matrix_obj.label} gradients'
        self.cmap = cmap
        # directory
        self.dir_path = self._get_dir_path()
        os.makedirs(self.dir_path, exist_ok=True)
        if os.path.exists(os.path.join(self.dir_path, 'gradients_surface.npz')):
            self._load()
        else:
            logging.info(f"Creating gradients in {self.dir_path}")
            self._create()
            self._save()
            if create_plots:
                self.plot_surface()
                self.plot_scatter()
                self.plot_scree()
                self.plot_reordered_matrix()

    def _create(self):
        """
        Creates the GradientMaps object and fits the data
        """
        if (self.matrix_obj.split_hem) & (self.hemi is None):
            # if the matrix obj is defined separately for L and R
            # (e.g., GD is regressed out of it) and this is the 
            # parent gradient where hemisphere should be concatenated
            # create separate gradient objects for L and R
            # TODO: This code is too complicated. Rewrite it
            split_gradients = {}
            for hemi in ['L', 'R']:
                split_gradients[hemi] = MicrostructuralCovarianceGradients(
                    **self._get_params(),
                    hemi = hemi
                ).parcellated_data
            # for each gradient (column) rescale the
            # values or R hem to L hem
            # Warning & TODO: the gradients from the
            # two hemispheres may have opposite directions
            # and I don't know how to fix it!
            for column in self.columns:
                scaler = MinMaxScaler(
                    (split_gradients['L'].loc[:, column].values.min(),
                    split_gradients['L'].loc[:, column].values.max())
                )
                split_gradients['R'].loc[:, column] = scaler.fit_transform(
                    split_gradients['R'].loc[:, column].values[:, np.newaxis]
                )[:, 0]
            # concatenate the hemispheres and label
            # the gradients
            gradients = np.concatenate([
                split_gradients['L'], split_gradients['R']
            ], axis=0)
            hem_parcels = helpers.get_hem_parcels(
                self.parcellation_name, 
                self.matrix_obj.matrix.index)
            self.parcellated_data = pd.DataFrame(
                gradients,
                index = hem_parcels['L'] + hem_parcels['R'],
                columns = self.columns
            )
            # project to surface
            self.surf_data = helpers.deparcellate(
                self.parcellated_data,
                self.matrix_obj.parcellation_name,
                downsampled = True
                )
        else:
            # initialize GradientMaps object
            self.gm = brainspace.gradient.GradientMaps(
                n_components=self._n_components_create, 
                approach=self._approach, 
                kernel=self._kernel, 
                random_state=912
                )
            # matrix pre-processing
            parcels = self.matrix_obj.matrix.index
            if (self.n_matrices > 1) & self._fair_sparsity:
                # enforce fair sparsity between fused matrices
                sparsity = None
                matrix = self.matrix_obj._make_sparse_fairly(sparsity=self._sparsity)
            else:
                sparsity = self._sparsity
                if self.hemi is not None:
                    # limit the matrix to the selected hemisphere
                    parcels = helpers.get_hem_parcels(
                        self.parcellation_name, 
                        parcels)[self.hemi]
                matrix = self.matrix_obj.matrix.loc[parcels, parcels].values
            # create the gradient
            self.gm.fit(matrix, sparsity=sparsity)
            self.lambdas = self.gm.lambdas_
            if self.parcellation_name is not None:
                # add parcel labels
                self.parcellated_data = pd.DataFrame(
                    self.gm.gradients_,
                    index=parcels,
                    columns=self.columns
                    )
                # project to surface
                self.surf_data = helpers.deparcellate(
                    self.parcellated_data,
                    self.matrix_obj.parcellation_name,
                    downsampled = True
                    )
            else:
                # bring back removed vertices with NaN input data
                # which are not included in the matrix or gradients
                self.surf_data = np.zeros((datasets.N_VERTICES_HEM_BB_ICO5*2, len(self.columns))) * np.NaN
                self.surf_data[self.matrix_obj.matrix.index, :] = self.gm.gradients_

    def _get_params(self):
        """
        Get the parameters used for intializing the
        object except hemi (for calling it recursively in case
        of split hemispheres gradients)
        """
        return dict(
            matrix_obj=self.matrix_obj, 
            n_components_create=self._n_components_create, 
            n_components_report=self._n_components_report,
            approach=self._approach, 
            kernel=self._kernel, 
            sparsity=self._sparsity, 
            fair_sparsity=self._fair_sparsity,
            cmap=self.cmap, 
            create_plots=False
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
        if self.hemi:
            sub_dir = os.path.join(sub_dir, f'hemi-{self.hemi}')
        return os.path.join(parent_dir, sub_dir)

    def _save(self):
        """
        Save the labeled gradients as .csv, surface maps as .npz, and
        lambdas as .txt
        """
        if self.parcellation_name is not None:
            self.parcellated_data.to_csv(
                os.path.join(self.dir_path, 'gradients_parcels.csv'), 
                index_label='parcel'
                )
        else:
            np.savez_compressed(
                os.path.join(self.dir_path,'gradients_surface.npz'),
                surface=self.surf_data
                )
        if hasattr(self, 'lambdas'):
            # the parent gradient object where split
            # gradients for the two hemispheres are joined
            # do not have lambdas 
            np.savetxt(
                os.path.join(self.dir_path,'lambdas.txt'),
                self.lambdas
                )

    def _load(self):
        """
        Load labeled gradients, surface map
        """
        if self.parcellation_name is not None:
            self.parcellated_data = pd.read_csv(
                os.path.join(self.dir_path, 'gradients_parcels.csv'),
                index_col='parcel')
            self.parcellated_data.columns = self.columns
            self.surf_data = helpers.deparcellate(
                self.parcellated_data,
                self.parcellation_name,
                downsampled = True
                )
        else:
            self.surf_data = np.load(os.path.join(self.dir_path, 'gradients_surface.npz'))['surface']
        if os.path.exists(os.path.join(self.dir_path, 'lambdas.txt')):
            # the parent gradient object where split
            # gradients for the two hemispheres are joined
            # do not have lambdas 
            self.lambdas = np.loadtxt(os.path.join(self.dir_path, 'lambdas.txt'))

    def plot_surface(self, layout_style='row', inflate=True):
        """
        Plots the gradients on the surface
        """
        # TODO: remove this as the parent object
        # has a very similar .plot function
        plots = []
        for gradient_num in range(1, self._n_components_report+1):
            plot = helpers.plot_surface(
                self.surf_data[:, gradient_num-1],
                filename=os.path.join(self.dir_path, f'surface_{layout_style}_G{gradient_num}'),
                layout_style=layout_style,
                inflate=inflate,
                cmap=self.cmap,
                cbar=True,
            )
            plots.append(plot)
        return plots

    def plot_scatter(self, remove_ticks=True):
        """
        Plot scatter plot of gradient values for G1 (x-axis) and G2 (y-axis) with
        colors representing G3

        Parameters
        ----------
        remove_ticks: (bool) remove ticks so that colorbars can replace them (manually)
        """
        if self.parcellation_name is None:
            logging.warn('Aborted plotting unparcellated scatter plot')
            return
        fig, ax = plt.subplots(figsize=(6,5))
        ax = sns.scatterplot(
            data=self.parcellated_data, 
            x=self.columns[0], # G1
            y=self.columns[1], # G2
            hue=self.columns[2], # G3
            palette=cmcrameri.cm.lajolla_r, # G3 cmap
            legend=False, ax=ax)
        if remove_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.savefig(
            os.path.join(self.dir_path,f'scatter.png'), 
            dpi=192)

    def plot_scree(self, normalize=False, mark_selected=True, cumulative=False):
        """
        Plot the lamdas

        Parameters
        ---------
        normalize: (bool) normalize the lambdas by sum
        """
        # plot the absolute/relative lambda values
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
        if mark_selected:
            # plot squares around the selected components
            x_selected = x[:self._n_components_report]
            y_selected = y[:self._n_components_report]
            ax.scatter(x_selected, y_selected, marker='s', s=200, facecolors='none', edgecolors='r')
        fig.savefig(
            os.path.join(self.dir_path,f'scree{"_normalized" if normalize else ""}.png'),
            dpi=192
            )
        if cumulative:
            # plot the cumulative variance explained
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
        if self.parcellation_name is None:
            logging.warn('Aborted plotting unparcellated reordered matrix')
            return
        if self.n_matrices == 1:
            # in unimodal case add ordered matrices by the first 3 gradients
            for g_idx in range(self._n_components_report):
                # sort parcels by gradient values
                sorted_parcels = self.parcellated_data.sort_values(by=self.columns[g_idx]).index
                # reorder matrix by sorted parcels
                reordered_matrix = self.matrix_obj.matrix.loc[sorted_parcels, sorted_parcels].values
                helpers.plot_matrix(
                    reordered_matrix,
                    os.path.join(self.dir_path, f'matrix_order-G{g_idx+1}'),
                    cmap=self.matrix_obj.cmap,
                    )
        else:
            for g_idx in range(self._n_components_report):
                # sort parcels by gradient values
                sorted_parcels = self.parcellated_data.sort_values(by=self.columns[g_idx]).index
                # split the matrix to square matrices, reorder each square matrix
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
        """
        super().__init__(*args, **kwargs)
        if 'thickness' in self.matrix_obj.input_type:
            self.cmap = 'viridis'
        else:
            self.cmap = 'Spectral_r'

    def plot_binned_profile(self, n_bins=10, palette='bigbrain'):
        """
        Plots the relative laminar thickness (TODO: and density) of `n_bins` bins of the top gradients
        """
        if self.matrix_obj.input_type != 'thickness':
            logging.warn(f"Plotting binned profiles is not implemented for input {self.matrix_obj.input_type}")
            return
        # specifiy the layer colors
        colors = datasets.LAYERS_COLORS.get(palette)
        if colors is None:
            colors = plt.cm.get_cmap(palette, 6).colors
        # loading and parcellating the laminar thickness
        if not hasattr(self.matrix_obj, '_input_data'):
            self.matrix_obj._load_input_data()
        if self.matrix_obj.dataset == 'bigbrain':
            laminar_thickness = self.matrix_obj._input_data
            parcellated_laminar_thickness = helpers.parcellate(laminar_thickness, self.matrix_obj.parcellation_name)
            parcellated_laminar_thickness = helpers.concat_hemispheres(parcellated_laminar_thickness, dropna=True)
            # re-normalize small deviations from sum=1 because of parcellation
            parcellated_laminar_thickness = parcellated_laminar_thickness.divide(parcellated_laminar_thickness.sum(axis=1), axis=0)
        elif self.matrix_obj.dataset == 'economo':
            parcellated_laminar_thickness = self.matrix_obj._parcellated_input_data
        for gradient_num in range(1, self._n_components_report+1):
            binned_parcels_laminar_thickness = parcellated_laminar_thickness.copy()
            binned_parcels_laminar_thickness['bin'] = pd.qcut(self.parcellated_data.iloc[:, gradient_num-1], n_bins)
            # calculate average laminar thickness at each bin
            bins_laminar_thickness = binned_parcels_laminar_thickness.groupby('bin').mean().reset_index(drop=True)
            bins_laminar_thickness.columns = [f'Layer {idx+1}' for idx in range(6)]
            # reverse the columns so that in the plot Layer 6 is at the bottom
            bins_laminar_thickness = bins_laminar_thickness[bins_laminar_thickness.columns[::-1]]
            # normalize to sum of 1 at each bin
            bins_laminar_thickness = bins_laminar_thickness.divide(bins_laminar_thickness.sum(axis=1), axis=0)
            # plot the relative thickness of layers 6 to 1
            # TODO: combine this with misc.py/plot_parcels_laminar_profile and put it in helpers.py
            # TODO: use BigBrain colormap
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                x = bins_laminar_thickness.index,
                height = bins_laminar_thickness['Layer 6'],
                width = 0.95,
                color=colors[-1],
                )
            for layer_num in range(5, 0, -1):
                ax.bar(
                    x = bins_laminar_thickness.index,
                    height = bins_laminar_thickness[f'Layer {layer_num}'],
                    width = 0.95,
                    bottom = bins_laminar_thickness.cumsum(axis=1)[f'Layer {layer_num+1}'],
                    color=colors[layer_num-1],
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlabel(f'G{gradient_num} bins')
            # ax.set_ylabel('Relative laminar thickness')
            ax.set_xlabel('')
            ax.set_ylabel('')
            for _, spine in ax.spines.items():
                spine.set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.dir_path, f'binned_profile_G{gradient_num}.png'), dpi=192)
            clfig = helpers.make_colorbar(
                self.parcellated_data.iloc[:,gradient_num-1].min(), 
                self.parcellated_data.iloc[:,gradient_num-1].max(),
                bins=10, 
                orientation='horizontal', figsize=(6,4))
            clfig.savefig(os.path.join(self.dir_path, f'binned_profile_G{gradient_num}_clbar.png'), dpi=192)

class CurvatureMap(ContCorticalSurface):
    """
    Map of bigbrain surface curvature
    """
    def __init__(self, downsampled=True):
        self.downsampled = downsampled
        self.surf_data = datasets.load_curvature_maps(
            downsampled=downsampled, 
            concatenate=True
            )[:, np.newaxis]
        self.columns = ['Curvature']
        self.label = 'Curvature'
        self.dir_path = os.path.join(OUTPUT_DIR, 'curvature')
    
    def effect_on_laminar_thickness(self, correct_curvature, nbins=20,
                             exc_regions='adysgranular', palette='bigbrain'):
        """
        Plot the laminar profile ordered by binned curvature
        to show how it differs for corrected and uncorrected 
        laminar thickness and plots a regplot on how deep layers
        ratio is associated with curvature
        """
        # load thickness data
        if correct_curvature is None:
            laminar_data = datasets.load_laminar_thickness(exc_regions=exc_regions)
        elif 'smooth' in correct_curvature:
            smooth_disc_radius = int(correct_curvature.split('-')[1])
            laminar_data = datasets.load_laminar_thickness(
                exc_regions=exc_regions,
                smooth_disc_radius=smooth_disc_radius
            )
        elif correct_curvature == 'regress':
            laminar_data = datasets.load_laminar_thickness(
                exc_regions=exc_regions,
                regress_out_curvature=True,
            )
        if laminar_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB:
            laminar_data = helpers.downsample(laminar_data)
        laminar_data = np.concatenate([laminar_data['L'], laminar_data['R']])
        # 1. Correlation with superficial laminar ratio
        superficial_ratio_obj = ContCorticalSurface(
            laminar_data[:, :3].sum(axis=1) / laminar_data.sum(axis=1), 
            columns = ['Superficial laminar thickness ratio'],
            label = f'Superficial laminar thickness ratio {correct_curvature}'
        )
        self.correlate(superficial_ratio_obj, parcellated=False)
        # 2. Plotting laminar profile per bin
        # convert to dataframe and rename columns
        laminar_data = pd.DataFrame(laminar_data,
                                columns = [f'Layer {idx+1}' for idx in range(6)]
                                )
        # reverse the columns so that in the plot Layer 6 is at the bottom
        laminar_data = laminar_data[laminar_data.columns[::-1]]
        # calculate the curvature bin
        laminar_data['bin'] = pd.qcut(pd.Series(self.surf_data[:, 0]), nbins)
        # average laminar thickness per bin
        laminar_data = laminar_data.groupby('bin').mean().reset_index(drop=True)
        # specifiy the layer colors
        colors = datasets.LAYERS_COLORS.get(palette)
        if colors is None:
            colors = plt.cm.get_cmap(palette, 6).colors
        # plot the relative thickness of layers 6 to 1
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(
            x = laminar_data.index,
            height = laminar_data['Layer 6'],
            width = 1,
            color=colors[-1],
            )
        for layer_num in range(5, 0, -1):
            ax.bar(
                x = laminar_data.index,
                height = laminar_data[f'Layer {layer_num}'],
                width = 1,
                bottom = laminar_data.cumsum(axis=1)[f'Layer {layer_num+1}'],
                color=colors[layer_num-1],
                )
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.dir_path, 
                f'laminar_profile_correct-{correct_curvature}_exc-{exc_regions}_nbins-{nbins}'),
            dpi=192)
        return ax

class GeodesicDistance(ContCorticalSurface):
    """
    Map of minimum geodesic distance from given seeds

    Depends on wb_command
    """
    def __init__(self, seed_parcellation='brodmann', 
                 seed_parcels=['BA17','BA1_3','BA41_42_52']):
        """
        Initializes geodesic distance surface from the center of `parcel`
        in `parcellation_name`
        """
        self.seed_parcellation = seed_parcellation
        self.seed_parcels = seed_parcels
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'distance',
            f'geodesic_{seed_parcellation}_{"-".join(seed_parcels)}'
        )
        os.makedirs(self.dir_path, exist_ok=True)
        self.file_path = os.path.join(self.dir_path, 'surf_data.npz')
        if os.path.exists(self.file_path):
            self.surf_data = np.load(self.file_path)['surf_data']
        else:
            self.surf_data = self._create()
            np.savez_compressed(self.file_path, surf_data=self.surf_data)
        self.label = f'Geodesic distance from {"-".join(seed_parcels)}'
        self.columns = [self.label]
        self.cmap = 'pink'
    
    def _create(self):
        """
        Creates the GD map from seed parcel using wb_command
        """
        meshes = datasets.load_downsampled_surface_paths('inflated')
        parcel_centers = helpers.get_parcel_center_indices(
            self.seed_parcellation, 
            downsampled=True
            )
        surf_data = []
        for seed_parcel in self.seed_parcels:
            curr_seed_surf = {}
            for hem in ['L', 'R']:
                # get the center vertex
                center_vertex = parcel_centers[hem][seed_parcel]
                # using wb_command get the GD map from the vertex and save it temporarily
                tmp_file_path = os.path.join(self.dir_path, f'{hem}.func.gii')
                cmdStr = f"{WB_PATH} -surface-geodesic-distance {meshes[hem]} {center_vertex} {tmp_file_path}"
                subprocess.run(cmdStr.split())
                curr_seed_surf[hem] = nilearn.surface.load_surf_data(tmp_file_path)
                # remove the tmp file
                os.remove(tmp_file_path)
            surf_data.append(
                np.concatenate([curr_seed_surf['L'], curr_seed_surf['R']])[:, np.newaxis]
            )
        surf_data = np.hstack(surf_data).min(axis=1)[:, np.newaxis]
        # get a maks of midline (from sjh parcellation)
        # and make the map NaN at midline
        midline_mask = np.isnan(
            helpers.deparcellate(
                helpers.parcellate(surf_data, 'sjh'), 
                'sjh', 
                downsampled=True
                )[:, 0])
        surf_data[midline_mask] = np.NaN
        return surf_data

class MyelinMap(ContCorticalSurface):
    """
    HCP S1200 group-averaged myelin (T1/T2) map transformed
    to bigbrain
    """
    def __init__(self, parcellation_name=None, exc_regions=None, downsampled=True):
        """
        Initializes the myelin map
        """
        self.parcellation_name = parcellation_name
        self.exc_regions = exc_regions
        self.surf_data = datasets.load_hcp1200_myelin_map(exc_regions, downsampled)
        self.columns = ['Myelin']
        self.label = 'HCP 1200 Myelin'
        self.cmap = 'pink_r'
        self.dir_path = os.path.join(OUTPUT_DIR, 'myelin')
        os.makedirs(self.dir_path, exist_ok=True)
        if self.parcellation_name:
            self.parcellated_data = helpers.parcellate(self.surf_data, self.parcellation_name)
            self.parcellated_data.columns = self.columns


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
        # load disorder maps from ENIGMA toolbox
        # (in DK parcellation)
        self.dir_path = os.path.join(OUTPUT_DIR, 'disease')
        os.makedirs(self.dir_path, exist_ok=True)
        self.label = 'Cortical thickness difference in diseases'
        self.parcellation_name = 'aparc'
        self.parcellated_data = datasets.load_disease_maps(psych_only=False, rename=True)
        # project it back to surface
        self.surf_data = helpers.deparcellate(self.parcellated_data, 'aparc', downsampled=True)
        self.columns = self.parcellated_data.columns

class EffectiveConnectivityMaps(ContCorticalSurface):
    """
    Maps of total afferent and efferent connectivity
    strength as well as their difference (hierarchy strength)
    """
    label = 'Effective connectivity maps'
    cmap = 'YlGnBu_r'
    def __init__(self, dataset='hcp'):
        """
        Loads the EC matrix and calculates the three
        measures of EC strength for each node and
        projects it to surface
        """
        self.ec = matrices.ConnectivityMatrix(
            'effective', 
            dataset=dataset, 
            exc_regions=None
            )
        self.parcellation_name = 'schaefer400'
        self.dir_path = os.path.join(self.ec.dir_path, 'maps')
        os.makedirs(self.dir_path, exist_ok=True)
        self._create()
    
    def _create(self):
        """
        Calculates the three measures of EC strength for each node and
        projects them to surface
        """
        afferent = self.ec.matrix.sum(axis=1)
        efferent = self.ec.matrix.sum(axis=0)
        hierarchy = efferent - afferent
        self.parcellated_data = pd.DataFrame(
            {
                'afferent': afferent,
                'efferent': efferent,
                'hierarchy': hierarchy,
            }
        )
        self.surf_data = helpers.deparcellate(
            self.parcellated_data,
            'schaefer400',
            downsampled=True
        )
        self.columns = self.parcellated_data.columns


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
        # parcellate the continous data
        parcellated_data = helpers.parcellate(surf_data, parcellation_name)
        parcellated_data.columns = columns
        # add the category of parcels to the parcellated data
        is_downsampled = (surf_data.shape[0] == datasets.N_VERTICES_HEM_BB_ICO5*2)
        parcellated_data.loc[:, self.label] = self.load_parcels_categories(parcellation_name, downsampled=is_downsampled)
        # exclude unwanted categories
        parcellated_data = parcellated_data[~parcellated_data[self.label].isin(self.excluded_categories)]
        parcellated_data[self.label] = parcellated_data[self.label].cat.remove_unused_categories()
        # remove NaNs (usually from surf_data since NaN is an unwanted category)
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
        # colorbar with the correct vmin and vmax
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
        #> assign each vertex to one of the 10 bins
        _, bin_edges = np.histogram(parcellated_data.loc[:,column], bins=nbins)
        parcellated_data[f'{column}_bin'] = np.digitize(parcellated_data.loc[:,column], bin_edges[:-1])
        #> calculate ratio of categories in each bin
        bins_categories_counts = (parcellated_data
                            .groupby([f'{column}_bin',self.label])
                            .size().unstack(fill_value=0))
        bins_categories_freq = bins_categories_counts.divide(bins_categories_counts.sum(axis=1), axis=0)
        #> plot stacked bars at each bin showing freq of the cortical categories
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

    def compare(self, other, other_columns=None, nbins=10):
        """
        Compares the difference of `other.surf_data` across `self.included_categories` and 
        plots it as raincloud or stacked bar plots

        Parameters
        ----------
        other: (ContCorticalSurface)
        other_columns: (list of str | None)
        nbins: (int)
            number of bins in the binned stacked bar plot
        """
        if other_columns is None:
            other_columns = other.columns
            surf_data = other.surf_data
        else:
            surf_data = pd.DataFrame(other.surf_data, columns=other.columns).loc[:, other_columns].values
        # parcellate data and specify cortical type of each parcel
        parcellated_data = self._parcellate_and_categorize(surf_data, other_columns, other.parcellation_name)
        # specify output dir
        out_dir = os.path.join(other.dir_path, f'association_{self.label.lower().replace(" ", "_")}')
        os.makedirs(out_dir, exist_ok=True)
        # investigate the association of gradient values and cortical categories (visually and statistically)
        anova_res_str = "ANOVA Results\n--------\n"
        for column in other_columns:
            # 1) Raincloud plot
            self._plot_raincloud(parcellated_data, column, out_dir)
            # 2) Binned stacked bar plot
            self._plot_binned_stacked_bar(parcellated_data, column, out_dir, nbins)
            # ANOVA
            anova_res_str += self._anova(parcellated_data, column, output='text')
        logging.info(anova_res_str)
        with open(os.path.join(out_dir, 'anova.txt'), 'w') as anova_res_file:
            anova_res_file.write(anova_res_str)
    
    def spin_compare(self, other, other_columns=None, n_perm=1000):
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
        # create downsampled bigbrain spin permutations
        assert n_perm <= 1000
        helpers.create_bigbrain_spin_permutations(n_perm=n_perm, is_downsampled=True)
        logging.info(f"Comparing surface data across {self.label} with spin test ({n_perm} permutations)")
        if other_columns is None:
            other_columns = other.columns
            surf_data = other.surf_data
        else:
            surf_data = pd.DataFrame(other.surf_data, columns=other.columns).loc[:, other_columns].values
        # split the other surface in two hemispheres
        split_surf_data = {
            'L': surf_data[:surf_data.shape[0]//2],
            'R': surf_data[surf_data.shape[0]//2:]
        }
        # load the spin permutation indices
        spin_indices = np.load(os.path.join(
            SRC_DIR, f'tpl-bigbrain_desc-spin_indices_downsampled_n-{n_perm}.npz'
            )) # n_perm * n_vert arrays for 'lh' and 'rh'
        # create the lh and rh surrogates and concatenate them
        surrogates = np.concatenate([
            split_surf_data['L'][spin_indices['lh']], 
            split_surf_data['R'][spin_indices['rh']]
            ], axis=1) # n_perm * n_vert * n_features
        # add the original surf_data at the beginning
        downsampled_surf_data_and_surrogates = np.concatenate([
            surf_data[np.newaxis, :, :],
            surrogates
            ], axis = 0)
        all_anova_results = {column:[] for column in other_columns}
        for perm in range(n_perm+1):
            logging.info(perm)
            # get the F stat and T stats for each permutation
            curr_data = downsampled_surf_data_and_surrogates[perm, :, :]
            curr_parcelated_data = self._parcellate_and_categorize(curr_data, other_columns, other.parcellation_name)
            for column in other_columns:
                all_anova_results[column].append(
                    self._anova(curr_parcelated_data, column, output='stats', force_posthocs=True)
                )
        spin_pvals = pd.DataFrame()
        for column in other_columns:
            # calculate the two-sided non-parametric p-values as the
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
    dir_path = os.path.join(OUTPUT_DIR, 'ctypes')
    def __init__(self, exc_regions='adysgranular', downsampled=True, parcellation_name=None):
        """
        Loads the map of cortical types
        """
        self.exc_regions = exc_regions
        self.parcellation_name = parcellation_name
        self.colors = sns.color_palette("RdYlGn_r", 6)
        if self.exc_regions == 'adysgranular':
            self.included_categories = ['EU1', 'EU2', 'EU3', 'KO']
            self.excluded_categories = [np.NaN, 'ALO', 'AG', 'DG']
            self.colors = self.colors[2:]
        elif self.exc_regions == 'allocortex':
            self.included_categories = ['AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO']
            self.excluded_categories = [np.NaN, 'ALO']
        else: # this will not happen normally
            self.included_categories = ['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO']
            self.excluded_categories = [np.NaN]
            self.colors = sns.color_palette("RdYlGn_r", 7)
        self.n_categories = len(self.included_categories)
        # load unparcellated surface data
        cortical_types_map = datasets.load_cortical_types(downsampled=downsampled)
        self.surf_data = cortical_types_map.cat.codes.values.reshape(-1, 1).astype('float')
        self.surf_data[cortical_types_map.isin(self.excluded_categories), 0] = np.NaN
        self.columns = ['Cortical Type']
        if self.parcellation_name:
            parcellated_cortical_types = datasets.load_cortical_types(self.parcellation_name)
            self.parcellated_data = parcellated_cortical_types[
                parcellated_cortical_types.isin(self.included_categories)
                ].cat.codes.to_frame()
            self.parcellated_data.columns = self.columns

    def load_parcels_categories(self, parcellation_name, downsampled):
        return datasets.load_cortical_types(parcellation_name, downsampled=downsampled)

    
class YeoNetworks(CatCorticalSurface):
    """
    Map of Yeo networks
    """
    label = 'Resting state network'
    def __init__(self, downsampled=True):
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
        yeo_map = datasets.load_yeo_map(downsampled=downsampled)
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

class ReceptorMap(ContCorticalSurface):
    """
    Map of neurotransmitter receptors based on PET or autoradiography
    data. 
    Source: Hansen 2021
    """
    def __init__(self, receptor, source, parcellation_name):
        """
        Initializes receptor map

        Parameters
        ----------
        receptor: (str) name of the receptor
            - NMDA
            - GABAa
        source: (str)
            - PET
            - autoradiography
        parcellation_name: (str)
            for autoradiography only aparc is available
        """
        # TODO: consider having separate options for GABAa and GABAa/BZ
        self.receptor = receptor
        self.source = source
        if self.source == 'autoradiography':
            assert parcellation_name == 'aparc', \
                'Autoradiography data only available in aparc'
        self.parcellation_name = parcellation_name
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'ei', self.source,
            f'{receptor}_parc-{parcellation_name}'
        )
        self.file_path = os.path.join(
            self.dir_path, f'parcellated_density_zscore.csv'
        )
        self.label = f'{receptor} density ({source})'
        if os.path.exists(self.file_path):
            self.parcellated_data = pd.read_csv(self.file_path, index_col='parcel')
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            if self.source == 'PET':
                self.parcellated_data = datasets.fetch_pet(self.parcellation_name, self.receptor)
            else:
                self.parcellated_data = datasets.fetch_autoradiography().loc[:, receptor].to_frame()
            self.parcellated_data.to_csv(self.file_path, index_label='parcel')
        # Pseudo-projection of volumetric data to surface via 
        # parcellation for plotting etc.
        self.surf_data = helpers.deparcellate(
            self.parcellated_data, 
            self.parcellation_name,
            downsampled = True
            )
        self.columns = self.parcellated_data.columns.tolist()
        CMAPS = {
            'NMDA': 'YlOrRd',
            'GABAa': 'YlGnBu'
        }
        self.cmap = CMAPS[self.receptor]
    
class NeuronTypeMaps(ContCorticalSurface):
    """
    Maps of aggregated expression of genes associated
    with excitatory and inhibitory neuron

    Reference: Seidlitz 2020 (https://www.nature.com/articles/s41467-020-17051-5)
    """
    def __init__(self, neuron_type, parcellation_name, discard_rh=True, paper='any'):
        """
        Creates/loads the neuron type gene expression map

        Parameters
        ---------
        neuron_type: (str)
            - Neuro-Ex
            - Neuro-In
        parcellation_name: (str)
        discard_rh: (bool)
            limit the map to the left hemisphere
            Note: For consistency with other functions the right
            hemisphere vertices/parcels are not removed but are set
            to NaN
        paper: (str)
            the paper to use as the source for list of genes
                - 'any': default
                - 'Lake'
                - 'Habib'
                - 'Li'
        """
        self.neuron_type = neuron_type
        self.parcellation_name = parcellation_name
        self.discard_rh = discard_rh
        self.paper = paper
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'ei', 'gene_expression',
            f'{neuron_type.lower()}_parc-{parcellation_name}'\
            f'_paper-{self.paper}'\
            + ('_rh_discarded' if discard_rh else '')
        )
        self.file_path = os.path.join(
            self.dir_path, f'parcellated_mean_expression.csv'
        )
        LABELS = {
            'Neuro-Ex': 'Excitatory neurons gene expression',
            'Neuro-In': 'Inhibitory neurons gene expression'
        }
        self.label = LABELS.get(self.neuron_type)
        self.columns = [self.label]
        CMAPS = {
            'Neuro-Ex': 'YlOrRd',
            'Neuro-In': 'YlGnBu'
        }
        self.cmap = CMAPS[self.neuron_type]
        if os.path.exists(self.file_path):
            self.parcellated_data = pd.read_csv(self.file_path, index_col='parcel')
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self.parcellated_data = self._create()
            self.parcellated_data.to_csv(self.file_path, index_label='parcel')
        self.parcellated_data.columns = self.columns
        self.surf_data = helpers.deparcellate(
            self.parcellated_data, 
            self.parcellation_name,
            downsampled = True
            )

    def _create(self):
        """
        Creates the neuron type gene expression maps by
        taking the average expression over all genes
        associated with the cell type
        """
        # load cell type gene list from Seidlitz2020
        seidlitz_lists = pd.read_csv(os.path.join(SRC_DIR, 'cell_types_genes_Seidlitz2020.csv'),delimiter=";")
        mask = (seidlitz_lists['Class'] == self.neuron_type)
        if self.paper != 'any':
            mask = mask & (seidlitz_lists['Paper'] == self.paper)
        gene_list = (
            seidlitz_lists[mask]
            .loc[:, 'Genes':].to_numpy().flatten()
        )
        self.gene_list = np.unique(gene_list[~pd.isnull(gene_list)]).tolist()
        cell_type_expression = datasets.fetch_aggregate_gene_expression(
            self.gene_list,
            self.parcellation_name,
            discard_rh = self.discard_rh,
            merge_donors = 'genes',
        )
        return cell_type_expression