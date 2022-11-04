import os
import itertools
import numpy as np
from numpy.ma.core import nonzero
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import brainspace.gradient
import scipy.stats
import ptitprince
import nibabel
import nilearn.surface
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import statsmodels.stats.multitest
import subprocess
from sklearn.cross_decomposition import PLSRegression
import abagen

import helpers, datasets, matrices

# specify directories
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
WB_PATH = os.path.join(cwd, '..', 'tools', 'workbench', 'bin_linux64', 'wb_command')


class CorticalSurface:
    """
    Generic class for common functions on cortical surfaces
    """
    # default attributes
    cmap = 'viridis'
    space = 'bigbrain'
    # TODO: specify required fields and methods
    def __init__(self, surf_data, columns, label, dir_path=None, 
                 cmap='viridis', parcellation_name=None):
        """
        Initialize object using any custom n-d surf_data. This will be overwritten
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
                ).dropna()
            self.parcellated_data.columns = self.columns

    def plot(self, columns=None, label_loc=None, cmap=None, save=False,
             plot_downsampled=False, inflate=False, **plotter_kwargs):
        """
        Plot the map on surface

        Parameters
        ---------
        columns: (list | None)
        label_loc: (str | None)
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
        surf_data = pd.DataFrame(self.surf_data, columns=self.columns)
        if columns is None:
            columns = self.columns
        plots = []
        for column in columns:
            # TODO: some of the following transformation never / hardly ever happen
            if plot_downsampled:
                if (self.parcellation_name is None):
                    if (self.surf_data.shape[0]==datasets.N_VERTICES_HEM_BB*2):
                        column_data = helpers.downsample(surf_data.loc[:, column].values)
                    else:
                        column_data = surf_data.loc[:, column].values
                else:
                    column_data = helpers.deparcellate(self.parcellated_data.loc[:, column], 
                        self.parcellation_name, space=self.space, downsampled=True)
            else:
                if (self.parcellation_name is None):
                    if (self.surf_data.shape[0]==datasets.N_VERTICES_HEM_BB_ICO5*2):
                        column_data = helpers.upsample(surf_data.loc[:, column].values)
                    else:
                        column_data = surf_data.loc[:, column].values
                else:
                    column_data = helpers.deparcellate(self.parcellated_data.loc[:, column], 
                        self.parcellation_name, space=self.space, downsampled=False)
            label_text = ({label_loc:[column]} if label_loc else None)
            if cmap is None: cmap = self.cmap
            plot = helpers.plot_surface(
                np.array(column_data),
                (os.path.join(self.dir_path, f'surface_{column}') if save else None),
                label_text = label_text,
                cmap = cmap,
                plot_downsampled = plot_downsampled,
                inflate = inflate,
                space = self.space,
                **plotter_kwargs    
            )
            plots.append(plot)
        if len(plots) == 1:
            return plots[0] # so that in notebook it is directly displayed
        else:
            return plots

class ContCorticalSurface(CorticalSurface):
    """
    General class for continous cortical surfaces
    """
    def correlate(self, other, parcellated=True, n_perm=1000,
                 x_columns=None, y_columns=None, parcellated_method='variogram',
                 fdr_axis=None, barplot=True, regplot=True, axis_off=False,
                 sort_barplot=False, stats_on_regplot=True,  save_files=False):
        """
        Calculate the correlation of surface maps with permutation test
        (with surrogates created using spins or variograms) and plot 
        the scatter plots

        Parameters
        ---------
        self, other: (CorticalSurface) with surf_data fields which are n_vert x n_features np.ndarrays
        parcellated: (bool)
            - True: correlate across parcels
            - False: correlate across vertices (uses spin test for permutation)
        parcellated_method: (str)
            - 'variogram' (default)
            - 'spin'
        n_perm: (int) number of permutations. Max: 1000
        x_columns, y_columns: (list of str) selected columns from self (x) and other (y)
        axis_off: (bool) turn the axis off from the plot
        sort_barplot: (bool) sort bar plot based on correlation values
        fdr_axis:
            - None: no FDR correction
            - 'x' or 'y': FDR correction along different columns in 'x' or 'y'

        Returns
        -------
        coefs, pvals: (pd.DataFrame) y_columns * x_columns
        """
        # TODO: add an option for removing outliers
        if parcellated:
            assert self.parcellation_name == other.parcellation_name
            assert self.parcellation_name is not None
        else:
            assert self.surf_data.shape[0] == other.surf_data.shape[0]
        if save_files:
            out_dir = os.path.join(
                self.dir_path, 
                'correlation_'\
                +('parcellated_' if parcellated else '')\
                + other.label.lower().replace(" ", "_")\
                + f'_nperm-{n_perm}'\
                +(f'_fdr-{fdr_axis}' if (fdr_axis is not None) else ''))
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
        # statistical test
        if parcellated:
            if parcellated_method == 'spin':
                print("Calculating correlations with spin test (parcellated)")
                coefs, pvals, coefs_null_dist =  helpers.spin_test_parcellated(
                    X = x_data, 
                    Y = y_data,
                    parcellation_name = self.parcellation_name,
                    n_perm = n_perm,
                    space = self.space
                    )
            else:
                print("Calculating correlations with variogram test (parcellated)")
                coefs, pvals, coefs_null_dist =  helpers.variogram_test(
                    X = x_data, 
                    Y = y_data,
                    parcellation_name = self.parcellation_name,
                    n_perm = n_perm,
                    surrogates_path = os.path.join(self.dir_path, f'variogram_surrogates_{"-".join(x_columns)}'),
                    space = self.space
                    )
        else:
            print("Calculating correlations with spin test")
            coefs, pvals, coefs_null_dist =  helpers.spin_test(
                surface_data_to_spin = x_data.values, 
                surface_data_target = y_data.values,
                n_perm=n_perm,
                is_downsampled=True
                )
            # convert to df
            coefs = pd.DataFrame(coefs)
            pvals = pd.DataFrame(pvals)
            coefs.columns = pvals.columns = x_columns
            coefs.index = pvals.index = y_columns
        if save_files:
            # save null distribution for future reference
            np.savez_compressed(
                os.path.join(out_dir, 'coefs_null_dist.npz'),
                coefs_null_dist=coefs_null_dist
            )
        # apply FDR if indicated
        if fdr_axis is not None:
            if fdr_axis == 'x':
                pvals = pvals.T
            for column in pvals.columns:
                _, pvals.loc[:, column] = statsmodels.stats.multitest.fdrcorrection(pvals.loc[:, column])
            if fdr_axis == 'x':
                pvals = pvals.T
        if save_files:
            # clean and save the results
            coefs.to_csv(os.path.join(out_dir, 'coefs.csv'))
            pvals.to_csv(os.path.join(out_dir, 'pvals.csv'))
        if barplot:
            # bar plots for association of each x_column with
            # all y_columns
            sns.set_style("ticks")
            if len(y_columns) > 1: # only makes sense if there are > 1 y_columns
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
                    fig, ax = plt.subplots(figsize=(3, 2), dpi=192)
                    sns.barplot(
                        data=curr_coefs.to_frame(), 
                        x=x_column, y=curr_coefs.index, 
                        palette=colors, edgecolor=".2", ax=ax)
                    ax.axvline(0, color='black')
                    sns.despine(offset=1, left=True, trim=False, ax=ax)
                    ax.set_xlim((-1, 1))
                    ax.set_ylabel('')
                    ax.set_xlabel(f'Correlation with {x_column}')
                    if save_files:
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(out_dir, f'barplot_{x_column.lower().replace(" ", "_")}.png'),
                            dpi=192)
        if regplot:
            # regression plots
            for x_column in x_columns:
                for y_column in y_columns:
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=192)
                    sns.regplot(
                        x=x_data.loc[:,x_column], 
                        y=y_data.loc[:,y_column],
                        scatter_kws=dict(
                            alpha=(0.8 if parcellated else 0.2), 
                            s=5, color='grey'),
                        line_kws=dict(color='red'),
                        ax=ax)
                    sns.despine(offset=10, trim=True, ax=ax)
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                    if stats_on_regplot:
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
                    if save_files:
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(
                                out_dir, 
                                f'{x_column.lower().replace(" ", "_")}-'\
                                + f'{y_column.lower().replace(" ", "_")}.png'),
                            dpi=192)
        return coefs, pvals

    def ahba_pls(self, columns, n_genes=500, n_components=1, brain_specific=False):
        """
        Performs PLS between columns of surface and AHBA gene expression data

        Parameters
        ---------
        columns: (list of str)
        n_genes: (int)
            number of genes with the highest absolute weight to select
        n_components: (int)
            number of PLS components
        brain_specific: (bool)
            limit the genes to brain-specific ones

        Returns
        -------
        top_genes: (list of dict)
            each element corresponds to a PLS component and includes
            list of top genes with 'pos' and 'neg' weights
        pls: (sklearn.cross_decomposition.PLSRegression)
            fitted pls model
        """
        ahba_df = datasets.fetch_ahba_data(self.parcellation_name, ibf_threshold=0.5, missing='centroids')['all']
        if brain_specific:
            brain_specific_genes_in_ahba = ahba_df.columns.intersection(abagen.fetch_gene_group('brain'))
            ahba_df = ahba_df.loc[:, brain_specific_genes_in_ahba]
        shared_parcels = ahba_df.index.intersection(self.parcellated_data.index)
        x = ahba_df.loc[shared_parcels]
        y = self.parcellated_data.loc[shared_parcels, columns]
        pls = PLSRegression(n_components=n_components)
        pls.fit(x, y)
        weights=pd.DataFrame(pls.x_weights_, index=ahba_df.columns)
        top_genes = []
        for i in range(n_components):
            curr_top_genes = weights.iloc[:, i].abs().sort_values(ascending=False).iloc[:n_genes].index.tolist()
            weights_top_genes = weights.iloc[:, i].loc[curr_top_genes]
            top_pos = weights_top_genes.loc[weights_top_genes >= 0].index.tolist()
            top_neg = weights_top_genes.loc[weights_top_genes < 0].index.tolist()
            top_genes.append({'pos': top_pos, 'neg': top_neg})
        return top_genes, pls
                    
class LaminarFeatures(ContCorticalSurface):
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
        self.label = 'Laminar features'
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
            self.parcellated_data = self.parcellated_data.dropna()
            # get only the valid parcels outside of exc_regions or midline
            # which also exist in self._parcellated_input_data
            valid_parcels = helpers.get_valid_parcels(
                self.parcellation_name, 
                self.exc_regions, 
                downsampled=True).intersection(self.parcellated_data.index)
            self.parcellated_data = self.parcellated_data.loc[valid_parcels]
    
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
        #> Superficial thickness ratio
        superficial_thickness_ratio = rel_laminar_thickness[:, :3].sum(axis=1)[:, np.newaxis]
        features.append(superficial_thickness_ratio)
        self.columns += ['Superficial laminar thickness ratio']
        #> Laminar densities
        laminar_density = datasets.load_laminar_density(exc_regions=self.exc_regions)
        laminar_density = np.concatenate(
            [laminar_density['L'], laminar_density['R']], axis=0
            )
        laminar_density = helpers.downsample(laminar_density)
        features.append(laminar_density)
        self.columns += [f'Layer {num} density' for num in range(1, 7)]
        #> concatenate all the features into a single array
        self.surf_data = np.hstack(features)

class Gradients(ContCorticalSurface):
    def __init__(self, matrix_obj, n_components_create=10, n_components_report=2,
                 approach='dm', kernel='normalized_angle', sparsity=0.9, fair_sparsity=True,
                 hemi=None, cmap='viridis'):
        """
        Gradients of a Matrix object

        Parameters
        ---------
        matrix: (Matrix)
        n_components_create: (int) number of components to calculate
        n_components_report: (int) number of first n components to report / plot / associate
        approach: (str) dimensionality reduction approach
        kernel: (str) affinity matrix calculation kernel
        sparsity: (int) proportion of smallest elements to zero-out for each row.
        fair_sparsity: (bool) whether to make joined matrices sparse fairly
        hemi: (str | None) limit the gradient to selected hemisphere ('L' or 'R')
        """
        self.matrix_obj = matrix_obj
        self.parcellation_name = self.matrix_obj.parcellation_name
        self._n_components_create = n_components_create
        self._n_components_report = n_components_report #TODO: determine this programmatically based on scree plot
        self._approach = approach
        self._kernel = kernel
        # self.n_matrices = self.matrix_obj.matrix.shape[1] // self.n_parcels
        if hasattr(self.matrix_obj, 'input_type'):
            self.n_matrices = len(self.matrix_obj.input_type.split('-'))
        else:
            self.n_matrices = 1
        self._sparsity = sparsity
        self._fair_sparsity = fair_sparsity
        self.hemi = hemi
        self.columns = [f'{self.matrix_obj.short_label} G{num}' \
                        for num in range(1, self._n_components_create+1)]
        self.label = f'{self.matrix_obj.label} gradients'
        self.cmap = cmap
        # directory
        self.dir_path = self._get_dir_path()
        os.makedirs(self.dir_path, exist_ok=True)
        # load the matrix if it's already created
        if os.path.exists(os.path.join(self.dir_path, 'gradients_surface.npz')):
            self._load()
            return
        # otherwise create it
        ## setting n_parcels down here because in the case of unparcellated
        ## matrix if the gradeint and matrix are already created I do not
        ## load the actual huge matrix because I don't need it, and therefore
        ## I cannot set the n_parcels in those cases
        self.n_parcels = self.matrix_obj.matrix.shape[0]
        self._create()
        self._save()

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
                split_gradients[hemi] = Gradients(
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

    def plot_surface(self, layout_style='row', inflate=False, plot_downsampled=False):
        """
        Plots the gradients on the surface
        """
        # TODO: remove this as the parent object
        # has a very similar .plot function
        plots = []
        for gradient_num in range(1, self._n_components_report+1):
            if plot_downsampled:
                surf_data = self.surf_data[:, gradient_num-1]
            else:
                if self.parcellation_name is None:
                    surf_data = helpers.upsample(self.surf_data[:, gradient_num-1])
                else:
                    surf_data = helpers.deparcellate(self.parcellated_data.iloc[:, gradient_num-1], self.parcellation_name)
            plot = helpers.plot_surface(
                surf_data,
                filename=os.path.join(self.dir_path, f'surface_{layout_style}_G{gradient_num}'),
                layout_style=layout_style,
                inflate=inflate,
                plot_downsampled=plot_downsampled,
                cmap=self.cmap,
                cbar=True,
            )
            plots.append(plot)
        return plots

    def plot_scatter(self, remove_ticks=True, g3_cmap='magma_r', save=False):
        """
        Plot scatter plot of gradient values for G1 (x-axis) and G2 (y-axis) with
        colors representing G3

        Parameters
        ----------
        remove_ticks: (bool) remove ticks so that colorbars can replace them (manually)
        """
        if self.parcellation_name is None:
            print('Aborted plotting unparcellated scatter plot')
            return
        fig, ax = plt.subplots(figsize=(6,5), dpi=192)
        ax = sns.scatterplot(
            data=self.parcellated_data, 
            x=self.columns[0], # G1
            y=self.columns[1], # G2
            hue=self.columns[2], # G3
            palette=g3_cmap, # G3 cmap
            legend=False, ax=ax)
        if remove_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if save:
            fig.savefig(
                os.path.join(self.dir_path,f'scatter.png'), 
                dpi=192)

    def plot_scree(self, normalize=False, mark_selected=True, cumulative=False, save=False):
        """
        Plot the lamdas

        Parameters
        ---------
        normalize: (bool) normalize the lambdas by sum
        """
        # plot the absolute/relative lambda values
        fig, ax = plt.subplots(figsize=(6, 4), dpi=192)
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
        if save:
            fig.savefig(
                os.path.join(self.dir_path,f'scree{"_normalized" if normalize else ""}.png'),
                dpi=192
                )
        if cumulative:
            # plot the cumulative variance explained
            fig, ax = plt.subplots(figsize=(6,4), dpi=192)
            y = (self.lambdas.cumsum() / self.lambdas.sum()) * 100
            print("Cumulative explained variance among the top gradients", y)
            ax.plot(
                x, y,
                linewidth = 0.5,
                marker='o',
                markersize = 2,
                color = 'grey',
                )
            ax.set_ylim([0, 105])
            if save:
                fig.savefig(
                    os.path.join(self.dir_path,'scree_cum.png'),
                    dpi=192
                    )

    def plot_reordered_matrix(self, save=False):
        """
        Plot the input matrix reordered by gradient values
        """
        if self.parcellation_name is None:
            print('Aborted plotting unparcellated reordered matrix')
            return
        if self.n_matrices == 1:
            # in unimodal case add ordered matrices by the first 3 gradients
            for g_idx in range(self._n_components_report):
                # sort parcels by gradient values
                sorted_parcels = self.parcellated_data.sort_values(by=self.columns[g_idx]).index
                # reorder matrix by sorted parcels
                reordered_matrix = self.matrix_obj.matrix.loc[sorted_parcels, sorted_parcels].values
                if save:
                    file_path = os.path.join(self.dir_path, f'matrix_order-G{g_idx+1}')
                else:
                    file_path = None
                helpers.plot_matrix(reordered_matrix, file_path, cmap=self.matrix_obj.cmap)
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
                if save:
                    file_path = os.path.join(self.dir_path, f'matrix_order-G{g_idx+1}')
                else:
                    file_path = None
                helpers.plot_matrix(reordered_split_matrices, file_path, cmap=self.matrix_obj.cmap)

class MicrostructuralCovarianceGradients(Gradients):
    def __init__(self, *args, **kwargs):
        """
        Gradients of microstructural covariance matrices (LTC, MPC and fused)
        """
        super().__init__(*args, **kwargs)
        if 'thickness' in self.matrix_obj.input_type:
            self.cmap = 'viridis'
        else:
            self.cmap = 'Spectral_r'
        # flip the gradients to match the default
        default_matrix = matrices.MicrostructuralCovarianceMatrix(self.matrix_obj.input_type)
        if self.matrix_obj.dir_path != default_matrix.dir_path:
            default_gradients = MicrostructuralCovarianceGradients(default_matrix)
            for g_idx in range(self.surf_data.shape[1]):
                corr = pd.DataFrame({
                    'current': self.surf_data[:, g_idx], 
                    'default': default_gradients.surf_data[:, 0]}
                    ).corr().iloc[0, 1]
                if corr < 0:
                    self.surf_data[:, g_idx] = - self.surf_data[:, g_idx]
                    if self.matrix_obj.parcellation_name is not None:
                        self.parcellated_data.iloc[:, g_idx] = - self.parcellated_data.iloc[:, g_idx]
        # flip the G1 if needed
        # default_gradient = MicrostructuralCovarianceGradients(
        #     matrices.MicrostructuralCovarianceMatrix(self.matrix_obj.input_type))
        # if self.matrix_obj.parcellation_name == 'schaefer400':
        #     self.parcellated_data.iloc[:, 0] = - self.parcellated_data.iloc[:, 0]
        #     self.surf_data[:, 0] = - self.surf_data[:, 0]

    def plot_binned_profile(self, n_bins=10, palette='bigbrain', use_laminar_density=True, zscore=False, save=False):
        """
        Plots the relative laminar thickness and density of `n_bins` 
        bins of the top gradients

        Parameters
        ----------
        n_bins: (int)
        palette: (str)
            color palette for laminar thickness 
            - bigbrain
            - wagstyl
        use_laminar_thickness: (bool)
            for fused gradient plots average density per
            layer instead of total depth density for all
            50 layer-agnostic points
        zscore: (bool)
            for density, zscore density values across parcel to remove
            brain-wide effects
        """
        # 1) load and parcellate density and/or thickness
        if 'density' in self.matrix_obj.input_type:
            # load and parcellate density profiles
            if self.matrix_obj.input_type == 'thickness-density':
                self.matrix_obj._create()
                mpc = self.matrix_obj.children['density']
            else:
                mpc = self.matrix_obj
            if not hasattr(mpc, '_input_data'):
                mpc._load_input_data()
            parcellated_density_profiles = helpers.parcellate(mpc._input_data, mpc.parcellation_name)
            parcellated_density_profiles = helpers.concat_hemispheres(parcellated_density_profiles, dropna=True)
            # normalize and invert values because brighter on the image means
            # lower density but we want to have higher values for regions with
            # higher density
            parcellated_density_profiles = parcellated_density_profiles / parcellated_density_profiles.values.max()
            parcellated_density_profiles = 1 - parcellated_density_profiles
            if zscore:
                parcellated_density_profiles = parcellated_density_profiles.apply(scipy.stats.zscore, axis=1)
        if 'thickness' in self.matrix_obj.input_type:
            if self.matrix_obj.input_type == 'thickness-density':
                self.matrix_obj._create()
                ltc = self.matrix_obj.children['thickness']
            else:
                ltc = self.matrix_obj
            if not hasattr(ltc, '_input_data'):
                ltc._load_input_data()
            parcellated_laminar_thickness = helpers.parcellate(ltc._input_data, ltc.parcellation_name)
            parcellated_laminar_thickness = helpers.concat_hemispheres(parcellated_laminar_thickness, dropna=True)
            # re-normalize small deviations from sum=1 because of parcellation
            if self.matrix_obj.relative:
                parcellated_laminar_thickness = parcellated_laminar_thickness.divide(parcellated_laminar_thickness.sum(axis=1), axis=0)
        if (self.matrix_obj.input_type == 'thickness-density') & use_laminar_density:
            laminar_density = datasets.load_laminar_density()
            parcellated_laminar_density = helpers.parcellate(laminar_density, mpc.parcellation_name)
            parcellated_laminar_density = helpers.concat_hemispheres(parcellated_laminar_density, dropna=True)
            parcellated_laminar_density = parcellated_laminar_density / parcellated_laminar_density.values.max()
            parcellated_laminar_density = 1 - parcellated_laminar_density
            if zscore:
                parcellated_laminar_density = parcellated_laminar_density.apply(scipy.stats.zscore, axis=1)
        # 2) plot profiles for each bin
        for gradient_num in range(1, self._n_components_report+1):
            # calculate average density and or thickness per gradient bin
            if 'thickness' in self.matrix_obj.input_type:
                binned_parcels_laminar_thickness = parcellated_laminar_thickness.copy()
                binned_parcels_laminar_thickness['bin'] = pd.qcut(self.parcellated_data.iloc[:, gradient_num-1], n_bins)
                # calculate average laminar thickness at each bin
                bins_laminar_thickness = binned_parcels_laminar_thickness.groupby('bin').mean().reset_index(drop=True)
                bins_laminar_thickness.columns = [f'Layer {idx+1}' for idx in range(6)]
                # reverse the columns so that in the plot Layer 6 is at the bottom
                bins_laminar_thickness = bins_laminar_thickness[bins_laminar_thickness.columns[::-1]]
                # normalize to sum of 1 at each bin
                if self.matrix_obj.relative:
                    bins_laminar_thickness = bins_laminar_thickness.divide(bins_laminar_thickness.sum(axis=1), axis=0)
            if 'density' in self.matrix_obj.input_type:
                # calculate average density per bin of gradient
                binned_parcels_density = parcellated_density_profiles.copy()
                binned_parcels_density['bin'] = pd.qcut(self.parcellated_data.iloc[:, gradient_num-1], n_bins)
                bins_density = binned_parcels_density.groupby('bin').mean().reset_index(drop=True)
                # (pseudo)upsample it to 250 points per bin
                bins_density = np.repeat(bins_density.values, 5, axis=1)
            if (self.matrix_obj.input_type == 'thickness-density') & use_laminar_density:
                # calculate average laminar density per bin of gradient
                binned_parcels_laminar_density = parcellated_laminar_density.copy()
                binned_parcels_laminar_density['bin'] = pd.qcut(self.parcellated_data.iloc[:, gradient_num-1], n_bins)
                bins_laminar_density = binned_parcels_laminar_density.groupby('bin').mean().reset_index(drop=True)
                bins_laminar_density.columns = [f'Layer {idx+1}' for idx in range(6)]
            # create the plots
            fig, ax = plt.subplots(figsize=(10, 7), dpi=192)
            if self.matrix_obj.input_type == 'density':
                sns.heatmap(bins_density.T, ax=ax, cmap='Blues', cbar=False)
            elif self.matrix_obj.input_type == 'thickness-density':
                # identify layer boundary markers:
                # for each bin identify 5 integers (0 to 249)
                # showing the layer boundary according to average
                # laminar thickness
                laminar_boundary_mask = pd.DataFrame(np.zeros((250, 10)))
                boundary_idx = (bins_laminar_thickness * 250).iloc[:, ::-1].cumsum(axis=1).astype('int') - 1
                for bin_idx, boundaries in boundary_idx.iterrows():
                    laminar_boundary_mask.loc[boundaries.values[:-1], bin_idx] = 1
                if use_laminar_density:
                    # convert the boundary mask to a segmentation, i.e.,
                    # asssigning the 250 points at each bin to layers 1 to 6
                    # based on their average relative thickness
                    bin_layer_mask = (laminar_boundary_mask.cumsum(axis=0) + 1).astype('int')
                    bins_thickness_density = bin_layer_mask.copy()
                    for bin_num in range(n_bins):
                        for layer_num in range(1, 7):
                            # for each bin and each layer replace
                            # the identifier of layer with the average
                            # density in that layer
                            bins_thickness_density.loc[
                                bins_thickness_density.loc[:, bin_num] == layer_num, 
                                bin_num
                            ] = bins_laminar_density.loc[bin_num, f'Layer {layer_num}']
                    sns.heatmap(bins_thickness_density, cmap='Greys')
                else:
                    # plot total length density
                    sns.heatmap(bins_density.T, ax=ax, cmap='Blues', cbar=False)                
                    ## set 0s to NaN so they are transparent on the heatmap
                    laminar_boundary_mask[laminar_boundary_mask==0] = np.NaN
                    # add the layer boundary markers to the axis
                    sns.heatmap(laminar_boundary_mask, ax=ax, cmap='Greys_r', cbar=False)
            elif self.matrix_obj.input_type == 'thickness':
                # specifiy the layer colors
                colors = datasets.LAYERS_COLORS.get(palette)
                if colors is None:
                    colors = plt.cm.get_cmap(palette, 6).colors
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
            # figure aesthetics and saving  
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.axis('off')
            if save:
                fig.tight_layout()
                fig.savefig(os.path.join(self.dir_path, f'binned_profile_G{gradient_num}.png'), dpi=192)
            clfig = helpers.make_colorbar(
                self.parcellated_data.iloc[:,gradient_num-1].min(), 
                self.parcellated_data.iloc[:,gradient_num-1].max(),
                bins=10, 
                cmap = self.cmap,
                orientation='horizontal', figsize=(6,4))
            if save:
                clfig.savefig(os.path.join(self.dir_path, f'binned_profile_G{gradient_num}_clbar.png'), dpi=192)

class CurvatureMap(ContCorticalSurface):
    def __init__(self, downsampled=True):
        """
        Map of bigbrain surface curvature
        """
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
        fig, ax = plt.subplots(figsize=(12, 4), dpi=192)
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
    def __init__(self, seed_parcellation='brodmann', 
                 seed_parcels=['BA17','BA1_3','BA41_42_52']):
        """
        Map of minimum geodesic distance from given seeds
        
        Parameters
        ----------
        seed_parcellation: (str)
            default is Brodmann
        seed_parcels: (list of str)
            default are primary sensory areas
        
        Note: It depends on wb_command
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

class Coordinates(ContCorticalSurface):
    def __init__(self, parcellation_name, space='bigbrain'):
        """
        Map of x, y and z coordinates of vertices
        """
        self.parcellation_name = parcellation_name
        self.space = space
        self.label = 'Coordinates'
        meshes = datasets.load_mesh_paths('orig', space=space, downsampled=True)
        self.surf_data = np.concatenate([
            nilearn.surface.load_surf_mesh(meshes['L']).coordinates,
            nilearn.surface.load_surf_mesh(meshes['R']).coordinates,
        ], axis=0)
        self.parcellated_data = helpers.parcellate(self.surf_data, self.parcellation_name, space=self.space)
        # convert left-right to medial-lateral
        self.parcellated_data.iloc[:, 0] = self.parcellated_data.iloc[:, 0].abs()
        self.columns = ['Medial-Lateral', 'Posterior-Anterior', 'Inferior-Superior']
        self.parcellated_data.columns = self.columns


class MacaqueHierarchy(ContCorticalSurface):
    def __init__(self):
        """
        Macaque laminar-based hierarchy based on Burt 2018
        (https://doi.org/10.1038/s41593-018-0195-0)
        """
        self.parcellation_name = 'M132'
        self.label = 'Laminar-based hierarchy'
        self.cmap = 'PiYG'
        self.space = 'yerkes'
        self.columns = [self.label]
        self.dir_path = os.path.join(OUTPUT_DIR, 'macaque_hierarchy')
        os.makedirs(self.dir_path, exist_ok=True)
        self.parcellated_data = datasets.load_macaque_hierarchy().to_frame()
        self.parcellated_data.columns = self.columns
        self.surf_data = helpers.deparcellate(self.parcellated_data, self.parcellation_name,space='yerkes')

class MacaqueSurface(ContCorticalSurface):
    def __init__(self, human_surf_obj, columns=None, parcellation_name='M132'):
        """
        Human gradient to transformed to macaque cortex

        Parameters
        ---------
        human_surf_obj: (Gradient)
        columns: (list)
        parcellation_name: (str)
            macaque parcellation name
        """
        self.human_surf_obj = human_surf_obj
        self.cmap = self.human_surf_obj.cmap
        self.label = self.human_surf_obj.label + ' macaque'
        self.parcellation_name = parcellation_name
        self.columns = columns
        self.space = 'yerkes'
        if self.columns is None:
            self.columns = self.human_surf_obj.columns
        self.space = 'yerkes'
        human_surfs_ico7 = pd.DataFrame(
            helpers.upsample(human_surf_obj.surf_data),
            columns = self.human_surf_obj.columns
        )
        self.dir_path = os.path.join(self.human_surf_obj.dir_path, 'macaque')
        os.makedirs(self.dir_path, exist_ok=True)
        self.surf_data = pd.DataFrame()
        for column in self.columns:
            column_str = column.replace(" ", "_")
            macaque_paths = {
                'L' : os.path.join(self.dir_path, f'tpl-yerkes_hemi-L_den-32k_desc-{column_str}.shape.gii'),
                'R' : os.path.join(self.dir_path, f'tpl-yerkes_hemi-R_den-32k_desc-{column_str}.shape.gii')
            }
            if not os.path.exists(macaque_paths['L']):
                lh_bb_path = os.path.join(self.human_surf_obj.dir_path, f'tpl-bigbrain_hemi-L_desc-{column_str}.shape.gii')
                rh_bb_path = os.path.join(self.human_surf_obj.dir_path, f'tpl-bigbrain_hemi-R_desc-{column_str}.shape.gii')
                helpers.write_gifti(lh_bb_path, 
                                    data=human_surfs_ico7.loc[:,column].iloc[:datasets.N_VERTICES_HEM_BB])
                helpers.write_gifti(rh_bb_path, 
                                    data=human_surfs_ico7.loc[:,column].iloc[datasets.N_VERTICES_HEM_BB:])
                helpers.surface_to_surface_transform(
                    in_dir = self.human_surf_obj.dir_path,
                    out_dir = self.dir_path,
                    in_lh = f'tpl-bigbrain_hemi-L_desc-{column_str}.shape.gii',
                    in_rh = f'tpl-bigbrain_hemi-L_desc-{column_str}.shape.gii',
                    in_space = 'bigbrain',
                    out_space = 'fs_LR',
                    desc = column.replace(' ', '_')
                )
                macaque_paths = helpers.human_to_macaque(
                    {
                        'L': os.path.join(self.dir_path, f'tpl-fs_LR_hemi-L_den-32k_desc-{column_str}.shape.gii'),
                        'R': os.path.join(self.dir_path, f'tpl-fs_LR_hemi-R_den-32k_desc-{column_str}.shape.gii'),
                    }, 
                    desc = column.replace(' ', '_'))
            self.surf_data.loc[:, column] = np.concatenate([
                nilearn.surface.load_surf_data(macaque_paths['L']),
                nilearn.surface.load_surf_data(macaque_paths['R'])
            ])
        self.surf_data = self.surf_data.values
        self.parcellated_data = helpers.parcellate(
            self.surf_data, 
            'M132', space='yerkes', 
            na_ratio_cutoff=0.5)
        self.parcellated_data.columns = self.columns
                
class MyelinMap(ContCorticalSurface):
    def __init__(self, parcellation_name=None, exc_regions=None, downsampled=True):
        """
        HCP S1200 group-averaged myelin (T1/T2) map transformed
        to bigbrain
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

class EffectiveConnectivityMaps(ContCorticalSurface):
    def __init__(self, dataset='mics'):
        """
        Maps of total afferent and efferent connectivity
        strength as well as their difference (hierarchy strength)

        Parameters
        --------
        dataset: (str)
            - 'mics' [default]
            - 'hcp'
        """
        self.ec = matrices.ConnectivityMatrix(
            'effective', 
            dataset=dataset, 
            exc_regions=None,
            exc_contra=False
            )
        self.parcellation_name = 'schaefer400'
        self.label = 'Effective connectivity maps'
        self.cmap = 'YlGnBu_r'
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


class NeuronalSubtypesMaps(ContCorticalSurface):
    def __init__(self, neuron_type, parcellation_name, 
                 discard_rh=True):
        """
        Aggregate gene expression pattern of excitatory and inhibitory 
        neuronal subtypes across brain regions using AHBA data and based on gene lists
        from Lake 2016 (https://doi.org/10.1126/science.aaf1204), similar to the
        approach used by Burt 2018 (https://doi.org/10.1038/s41593-018-0195-0)

        Parameters
        ---------
        neuron_type: (str)
            - exc
            - inh
        parcellation_name: (str)
        discard_rh: (bool)
            limit the map to the left hemisphere
            Note: For consistency with other functions the right
            hemisphere vertices/parcels are not removed but are set
            to NaN
        """
        self.neuron_type = neuron_type
        self.parcellation_name = parcellation_name
        self.discard_rh = discard_rh
        LABELS = {
            'exc': 'Excitatory neuron subtypes gene expression maps',
            'inh': 'Inhibitory neuron subtypes gene expression maps',
        }
        self.label = LABELS.get(self.neuron_type)
        CMAPS = {
            'exc': 'YlOrRd',
            'inh': 'YlGnBu'
        }
        self.cmap = CMAPS[self.neuron_type]
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'ei',
            (f'{neuron_type}'
            + f'_parc-{self.parcellation_name}'
            + ('_rh_discarded' if discard_rh else '')
            ))
        os.makedirs(self.dir_path, exist_ok=True)
        self._create()

    def _create(self):
        """
        Loads the gene lists, aggregates their regional expression,
        and projects to surface
        """
        subtypes_genes = pd.read_csv(
            os.path.join(
                SRC_DIR, f'{self.neuron_type}_subtypes_genes_Lake2016.csv'
            ), delimiter=";", decimal=",").dropna()
        self.parcellated_data = pd.DataFrame()
        for subtype, subtype_df in subtypes_genes.groupby('cluster'):
            print(subtype)
            subtype_genes = subtype_df.set_index('Gene').loc[:, subtype]
            self.parcellated_data.loc[:, subtype] = datasets.fetch_aggregate_gene_expression(
                subtype_genes,
                parcellation_name = self.parcellation_name, 
                discard_rh = self.discard_rh,
                merge_donors = 'genes',
                ibf_threshold=0.25,
                )
        self.parcellated_data = self.parcellated_data.dropna()
        self.surf_data = helpers.deparcellate(
            self.parcellated_data, self.parcellation_name, downsampled=True
            )
        self.columns = self.parcellated_data.columns




########################
# Categorical surfaces #
######################## 

class CatCorticalSurface(CorticalSurface):
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
        parcellated_data.loc[:, self.label] = self._load_parcels_categories(parcellation_name, downsampled=is_downsampled)
        # exclude unwanted categories
        parcellated_data = parcellated_data[~parcellated_data[self.label].isin(self.excluded_categories)]
        parcellated_data[self.label] = parcellated_data[self.label].cat.remove_unused_categories()
        # remove NaNs (usually from surf_data since NaN is an unwanted category)
        parcellated_data = parcellated_data.dropna()
        return parcellated_data

    def _plot_raincloud(self, parcellated_data, column, out_dir=None):
        """
        Plots the raincloud of the `column` from `parcellated_data`
        across `self.included_categories` and saves it in `out_dir`
        """
        fig, ax = plt.subplots(figsize=(6, 4), dpi=192)
        ax = ptitprince.RainCloud(
            data=parcellated_data, 
            y=column,
            x=self.label,
            palette=self.colors,
            bw = 0.2, width_viol = 1, 
            orient = 'h', move = 0.2, alpha = 0.4,
            ax=ax)
        sns.despine(ax=ax, offset=10, trim=True)
        # colorbar with the correct vmin and vmax
        clbar_fig = helpers.make_colorbar(
            ax.get_xticks()[0], ax.get_xticks()[-1], 
            figsize=(4, 4), orientation='horizontal')
        clbar_fig.tight_layout()
        if out_dir:
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{column}_raincloud'), dpi=192)
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
        fig, ax = plt.subplots(figsize=(6, 4), dpi=192)
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
        clfig = helpers.make_colorbar(
            parcellated_data[column].min(), parcellated_data[column].max(),
            bins=nbins, orientation='horizontal', figsize=(6,4))
        if out_dir:
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{column}_stacked_bar'), dpi=192)
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

    def compare(self, other, other_columns=None, nbins=10, plot_type='both', save=False):
        """
        Compares the difference of `other.surf_data` across `self.included_categories` and 
        plots it as raincloud or stacked bar plots

        Parameters
        ----------
        other: (ContCorticalSurface)
        other_columns: (list of str | None)
        nbins: (int)
            number of bins in the binned stacked bar plot
        plot_type: (str)
            - 'both'
            - 'raincloud'
            - 'stacked_bar'
        save: (bool)
        """
        assert self.parcellation_name is not None
        assert self.parcellation_name == other.parcellation_name
        if other_columns is None:
            other_columns = other.columns
        parcellated_data = pd.concat([
            other.parcellated_data.loc[:, other_columns], # continous surface
            self.parcellated_data # categories
            ], axis=1).dropna()
        if save:
            # specify output dir
            out_dir = os.path.join(other.dir_path, f'association_{self.label.lower().replace(" ", "_")}')
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = None
        # investigate the association of gradient values and cortical categories (visually and statistically)
        anova_res_str = "ANOVA Results\n--------\n"
        for column in other_columns:
            # 1) Raincloud plot
            if plot_type in ['both', 'raincloud']:
                self._plot_raincloud(parcellated_data, column, out_dir)
            # 2) Binned stacked bar plot
            if plot_type in ['both', 'stacked_bar']:
                self._plot_binned_stacked_bar(parcellated_data, column, out_dir, nbins)
            # ANOVA
            anova_res_str += self._anova(parcellated_data, column, output='text')
        print(anova_res_str)
        if out_dir:
            with open(os.path.join(out_dir, 'anova.txt'), 'w') as anova_res_file:
                anova_res_file.write(anova_res_str)
    
class CorticalTypes(CatCorticalSurface):
    def __init__(self, exc_regions='adysgranular', downsampled=True, parcellation_name=None):
        """
        Map of cortical types

        Parameters
        ---------
        exc_regions: (None | str)
            - 'adysgranular' [default]
            - 'allocortex'
            - None
        downsampled: (bool)
        parcellation_name: (str)
        """
        self.exc_regions = exc_regions
        self.parcellation_name = parcellation_name
        self.label = 'Cortical Type'
        self.short_label = 'ctypes'
        self.cmap = 'RdYlGn_r' # for surface map
        self.colors = sns.color_palette("RdYlGn_r", 6) # for raincloud / barplot
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
        self.dir_path = os.path.join(OUTPUT_DIR, 'ctypes')
        os.makedirs(self.dir_path, exist_ok=True)
        # load unparcellated surface data
        cortical_types_map = datasets.load_cortical_types(downsampled=downsampled)
        self.surf_data = cortical_types_map.cat.codes.values.reshape(-1, 1).astype('float')
        self.surf_data[cortical_types_map.isin(self.excluded_categories), 0] = np.NaN
        self.columns = ['Cortical Type']
        if self.parcellation_name:
            parcellated_cortical_types = datasets.load_cortical_types(self.parcellation_name)
            self.parcellated_data = parcellated_cortical_types[
                parcellated_cortical_types.isin(self.included_categories)
                ].cat.remove_unused_categories().to_frame()
            self.parcellated_data.columns = self.columns

    def _load_parcels_categories(self, parcellation_name, downsampled):
        return datasets.load_cortical_types(parcellation_name, downsampled=downsampled)
    
class YeoNetworks(CatCorticalSurface):
    """
    Map of Yeo networks
    """
    def __init__(self, parcellation_name=None, downsampled=True):
        """
        Loads the map of Yeo networks
        """
        self.colors = self._load_colormap()
        self.excluded_categories = [np.NaN, 'None']
        self.included_categories = [
            'Visual', 'Somatomotor', 'Dorsal attention', 
            'Ventral attention', 'Limbic', 'Frontoparietal', 'Default'
            ]
        self.parcellation_name = parcellation_name
        self.label = 'Resting state network'
        self.short_label = 'rsn'
        self.n_categories = len(self.included_categories)
        if self.parcellation_name is not None:
            self.parcellated_data = datasets.load_yeo_map(self.parcellation_name, downsampled=downsampled).rename(self.label)
            self.parcellated_data = (
                self.parcellated_data
                .loc[~self.parcellated_data.isin(self.excluded_categories)]
                .cat.remove_unused_categories().to_frame()
            )
            self.surf_data = helpers.deparcellate(self.parcellated_data.iloc[:, 0].cat.codes, self.parcellation_name)
        else:
            # load unparcellated surface data
            yeo_map = datasets.load_yeo_map(downsampled=downsampled)
            self.surf_data = yeo_map.cat.codes.values.reshape(-1, 1).astype('float')
            self.surf_data[yeo_map.isin(self.excluded_categories), 0] = np.NaN
        self.columns = [self.label]

    def _load_parcels_categories(self, parcellation_name, downsampled):
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

class MicrostructuralClusters(CatCorticalSurface):
    """
    Clusters of cortical surface based on input
    microstructural features (laminar thickness, density)
    """
    label = 'Cluster'
    def __init__(self, matrix_obj):
        # TODO: this should not depend on the matrix objects
        # and should be able to accept its own parameters
        self.matrix_obj = matrix_obj
        self.parcellation_name = self.matrix_obj.parcellation_name
        self.dir_path = os.path.join(matrix_obj.dir_path, 'clustering')
        os.makedirs(self.dir_path, exist_ok=True)
        self._create()
        self.included_categories = np.unique(self.parcellated_data).tolist()
        self.excluded_categories = []
        self.surf_data = helpers.deparcellate(self.parcellated_data, 'sjh')
        self.cmap = 'Blues'
        self.colors = sns.color_palette(self.cmap, self.n_categories)
        self.columns = [self.label]

    def _create(self, random_state=921):
        # get the input data which involves rerunning
        # the ._load_input_data method of matrix object
        self.matrix_obj._load_input_data()
        model = KMeans()
        # identify the optimal number of clusters
        visualizer = KElbowVisualizer(model, k=(2,10), timings= False)
        visualizer.fit(self.matrix_obj._parcellated_input_data)
        visualizer.show(outpath=os.path.join(self.dir_path, 'optimal_n_clusters.png'))
        self.n_categories = visualizer.elbow_value_
        self.model = KMeans(n_clusters=self.n_categories, random_state=random_state)
        self.model.fit(self.matrix_obj._parcellated_input_data)
        self.parcellated_data = (
            pd.Series(
                self.model.predict(self.matrix_obj._parcellated_input_data) + 1, 
                index=self.matrix_obj._parcellated_input_data.index)
            .rename(self.label)
            .astype('category')
        )

    def plot_laminar_profiles(self, save=False):
        """
        Plot the thickness profiles for the clusters
        """
        assert self.matrix_obj.input_type == 'thickness'
        laminar_data = self.matrix_obj._parcellated_input_data.copy()
        laminar_data.columns = [f'Layer {idx+1}' for idx in range(6)]
        laminar_data = laminar_data.iloc[:, ::-1]
        laminar_data['km_cluster'] = self.parcellated_data.astype('int')
        laminar_data = laminar_data.sort_values('km_cluster').reset_index(drop=True)
        colors = datasets.LAYERS_COLORS['bigbrain']
        fig, ax = plt.subplots(figsize=(12, 4), dpi=192)
        # plot cluster identifiers at the bottom
        cluster_identifier_height = 0.05
        for cluster_id in range(1, self.n_categories+1):
            ax.bar(
                x = laminar_data[laminar_data['km_cluster']==cluster_id].index, 
                height = cluster_identifier_height, 
                width = 1, 
                color = self.colors[cluster_id-1])
        # plot laminar profiles
        ax.bar(
            x = laminar_data.index,
            height = laminar_data['Layer 6'],
            bottom = cluster_identifier_height,
            width = 1,
            color=colors[-1],
            )
        for layer_num in range(5, 0, -1):
            ax.bar(
                x = laminar_data.index,
                height = laminar_data[f'Layer {layer_num}'],
                width = 1,
                bottom = laminar_data.cumsum(axis=1)[f'Layer {layer_num+1}'] + cluster_identifier_height,
                color=colors[layer_num-1],
                )
        ax.vlines(
            laminar_data['km_cluster'][(laminar_data['km_cluster'].diff() == 1)].index, 
            ymin=0, 
            ymax=1+cluster_identifier_height,
            color='grey')
        ax.axis('off')
        if save:
            fig.tight_layout()
            fig.savefig(os.path.join(self.dir_path, 'laminar_profile.png'), dpi=192)
        return ax