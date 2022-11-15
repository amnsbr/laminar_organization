import os
import numpy as np
import nilearn.surface
import sklearn as sk
import scipy.stats
import subprocess
import nibabel
import scipy.spatial.distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multitest
import statsmodels.stats.anova
import statsmodels.api as sm
import cmcrameri.cm # color maps
import PIL
import abagen

import helpers, datasets, surfaces

# specify directories
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')
MICAPIPE_BASE = os.path.join(cwd, '..', 'tools', 'micapipe')
wbPath = os.path.join(cwd, '..', 'tools', 'workbench', 'bin_linux64', 'wb_command')

class Matrix:
    """
    Generic class for matrices
    """
    split_hem = False # set to False by default and only change for GD and SC
    cmap = 'rocket' # default cmap
    def __init__(self, matrix, parcellation_name, label, dir_path=None, short_label=None, cmap=None):
        """
        Initialize any custom matrix
        """
        self.matrix = matrix
        self.parcellation_name = parcellation_name
        self.label = label
        self.short_label = short_label
        self.short_label = self.label
        self.cmap = cmap
        self.dir_path = dir_path
        if self.dir_path:
            os.makedirs(self.dir_path, exist_ok=True)
            self.file_path = os.path.join(self.dir_path, 'matrix')
            self._save()

    def _save(self):
        """
        Save the matrix to a .npz file
        """
        # reducing the filesize by saving it as
        # compressed and changing values type to float16
        # Warning: this makes the matrix unsymmetric!
        np.savez_compressed(
            self.file_path+'.npz',
            matrix = self.matrix.values.astype('float16'),
            parcels = self.matrix.index.values
        )

    def _load(self, make_symmetric=True):
        """
        Loads a matrix created before

        Parameters
        ----------
        make_symmetric: (bool)
            because of the way matrices are saved
            sometimes float values are slightly
            modified and this makes the matrix
            non-symmetric and can cause problems with
            e.g., GD matrix in variogram surrogates
        """
        print(f"Loading the matrix from {self.file_path}.npz")
        npz = np.load(self.file_path+'.npz', allow_pickle=True)
        parcels = npz['parcels'].tolist()
        matrix = npz['matrix'].astype('float64') # many functions have problems with float16
        n_sq_matrix = matrix.shape[1] // matrix.shape[0]
        if make_symmetric:
            matrices = []
            for idx in range(n_sq_matrix):
                curr_matrix = matrix[:, matrix.shape[0]*idx:matrix.shape[0]*(idx+1)]
                curr_matrix = np.maximum(curr_matrix, curr_matrix.T)
                matrices.append(curr_matrix)
            matrix = np.hstack(matrices)
        self.matrix = pd.DataFrame(
            matrix, 
            index=parcels, 
            columns=parcels * n_sq_matrix)

    def _remove_parcels(self, exc_regions):
        """
        Removes the allocortex +/- adysgranular parcels from the matrix (after the matrix is created)
        
        Note: This should be used for the matrices other than MicrostructuralCovarianceMatrix
        as in that case the adysgranular regions are removed from the input data (this 
        is important when using partial correlation)

        exc_regions: (str)
            - allocortex: excludes allocortex
            - adysgranular: excludes allocortex + adysgranular regions
        """
        # get valid parcels based on exc_regions
        valid_parcels = helpers.get_valid_parcels(self.parcellation_name, exc_regions, downsampled=True)
        # get subset of valid parcels that exist in the matrix
        valid_parcels = valid_parcels.intersection(self.matrix.index).tolist()
        return self.matrix.loc[valid_parcels, valid_parcels]

    def plot(self, vrange=(0.025, 0.975), save=False):
        """
        Plot the matrix as heatmap
        """
        # plot the matrix
        helpers.plot_matrix(
            self.matrix.values,
            (self.file_path if save else None),
            cmap=self.cmap,
            vrange=vrange
            )

    def correlate_edge_wise(self, other, test='pearson', test_approach='spin', 
            n_perm=1000, plot_regplot=True, plot_half_matrices=False, 
            regress_out_gd=False, figsize=(6, 4), axis_off=False,
            stats_on_plot=True, half_matrix_vrange=(0.025, 0.975),
            save_files=False, verbose=True):
        """
        Calculates and plots the correlation between the edges of two matrices
        self and other which are assumed to be square and symmetric. The correlation
        is calculated for lower triangle (excluding the diagonal)

        Paremeters
        ---------
        self, other: (Matrix) with the field .matrix which is a pd.DataFrame (n_parc x n_parc)
        test: (str)
            - pearson
            - spearman
        test_approach: (str)
            - spin: create null distribution by rotating parcels
            - shuffle: create null distribution by shuffling parcels
            - param: parameteric
        n_perm: (int)
            number of permutations in case test_approach other than `param` is selected
        plot_half_matrices: (bool) plot upper/lower half of each matrix for using in the paper
        """
        # make sure they have the same parcellation and mask
        assert self.parcellation_name == other.parcellation_name
        # match the matrices in the order and selection of parcels
        # + convert them to np.ndarray
        if regress_out_gd:
            gd = DistanceMatrix(self.parcellation_name)
            X = gd.regress_out(self, save_plot=False)
            Y = gd.regress_out(other, save_plot=False)
        else:
            X = self.matrix
            Y = other.matrix
        shared_parcels = X.index.intersection(Y.index)
        X = X.loc[shared_parcels, shared_parcels]
        Y = Y.loc[shared_parcels, shared_parcels]
        # get the index for lower triangle
        tril_index = np.tril_indices_from(X.values, -1)
        x = X.values[tril_index]
        y = Y.values[tril_index]
        # remove NaNs (e.g. interhemispheric pairs)
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        # correlation
        if test == 'pearson':
            coef, p_val = scipy.stats.pearsonr(x, y)
        else:
            coef, p_val = scipy.stats.spearmanr(x, y)
        if test_approach != 'param':
            surrogate_matrices = self.get_surrogates(method=test_approach, n_perm=n_perm)
            null_dist = np.zeros(n_perm)
            for i, surrogate in enumerate(surrogate_matrices):
                null_dist[i], _ = surrogate.correlate_edge_wise(other, test=test, test_approach='param',
                    plot_regplot=False, save_files=False, verbose=False)
            p_val = (np.abs(null_dist) >= np.abs(coef)).mean()
        res_str = f"{test.title()} correlation with {other.label}\nCoef: {coef}; p-value ({test_approach}): {p_val}"
        if verbose:
            print(res_str)
        if save_files:
            out_path = os.path.join(self.dir_path, f'correlation_{other.label.lower().replace(" ", "_")}')
            with open(out_path+'.txt', 'w') as res_file:
                res_file.write(res_str)
        if plot_regplot:
            # plotting
            fig, ax = plt.subplots(1, figsize=figsize, dpi=192)
            ax.hexbin(x, y, cmap='gist_heat_r')
            sns.regplot(
                x = x, y = y, 
                ax=ax, ci=None, scatter=False, 
                color='black', line_kws=dict(alpha=0.6)
                )
            # ax.set_xlim((np.quantile(x, 0.025), np.quantile(x, 0.975)))
            # ax.set_ylim((np.quantile(y, 0.025), np.quantile(y, 0.975)))
            if stats_on_plot:
                # add rho on the figure
                text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
                text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.90
                if test == 'pearson':
                    text = f'r = {coef:.2f}; p = {p_val:.2f}'
                else:
                    text = f'rho = {coef:.2f}; p = {p_val:.2f}'
                ax.text(text_x, text_y, text,
                        color='black', size=12,
                        multialignment='left')
            ax.set_xlabel(self.label)
            ax.set_ylabel(other.label)
            if axis_off:
                ax.axis('off')
            if save_files:
                fig.tight_layout()
                fig.savefig(out_path+'.png', dpi=192)
        if plot_half_matrices:
            # plotting half-half matrix
            uhalf_X = X.values
            uhalf_X[np.tril_indices_from(X, 0)] = np.NaN
            lhalf_Y = Y.values
            lhalf_Y[np.triu_indices_from(Y, 0)] = np.NaN
            helpers.plot_matrix(
                uhalf_X, 
                (os.path.join(f'{out_path}_{self.label}_uhalf') if save_files else None),
                cmap = self.cmap,
                vrange = half_matrix_vrange
                )
            helpers.plot_matrix(
                lhalf_Y, 
                (os.path.join(f'{out_path}_{other.label}_lhalf') if save_files else None),
                cmap = other.cmap,
                vrange = half_matrix_vrange
                ) 
        return coef, p_val       

    def correlate_node_wise(self, other, test='pearson', plot=True, 
            plot_sig=False, plot_layout='grid', save=False):
        """
        Calculates the correlation of matrix with another matrix at each row (node),
        projects the node-wise Spearman's rho values to the surface and saves and plots it. 
        Matrices are assumed to be square and symmetric.

        Paremeters
        ---------
        self, other: (Matrix) with the field .matrix which is a pd.DataFrame (n_parc x n_parc)
        test: (str)
            - pearson
            - spearman
        plot: (bool) plot the surface map of FDR-corrected correlation coefficients
        plot_sig: (bool) only include correlation of parcels which are significantly
                        correlated after FDR correction.
                        Set to False by default as the significance is highly dependent
                        on the number of parcels
        plot_layout: (str) 'grid' or 'row'
        """
        # make sure they have the same parcellation and mask
        assert self.parcellation_name == other.parcellation_name
        # match the matrices in the order and selection of parcels
        # + convert them to np.ndarray
        shared_parcels = self.matrix.index.intersection(other.matrix.index)
        X = self.matrix.loc[shared_parcels, shared_parcels]
        Y = other.matrix.loc[shared_parcels, shared_parcels]
        # if L and R should be investigated separately
        # make interhemispheric pairs of the lower triangle
        # NaN so it could be then removed
        # it may be already NaN in these matrices
        X = X.values
        Y = Y.values
        # calculate the correlation at each row (node)
        node_coefs = pd.Series(np.empty(X.shape[0]), index=shared_parcels)
        node_pvals = pd.Series(np.empty(X.shape[0]), index=shared_parcels)
        for row_idx in range(X.shape[0]):
            row_x = X[row_idx, :]
            row_y = Y[row_idx, :]
            # remove NaNs 
            row_mask = ~(np.isnan(row_x) | np.isnan(row_y)) 
            row_x = row_x[row_mask]
            row_y = row_y[row_mask]
            # calculate correlation
            if test == 'pearson':
                node_coefs.iloc[row_idx], node_pvals.iloc[row_idx] = scipy.stats.pearsonr(row_x, row_y)
            else:
                node_coefs.iloc[row_idx], node_pvals.iloc[row_idx] = scipy.stats.spearmanr(row_x, row_y)
        # FDR correction
        _, node_pvals_fdr = statsmodels.stats.multitest.fdrcorrection(node_pvals)
        node_pvals_fdr = pd.Series(node_pvals_fdr, index=shared_parcels)
        node_coefs_sig = node_coefs.copy()
        node_coefs_sig.loc[node_pvals_fdr >= 0.05] = 0
        # project to surface
        node_coefs_surface = helpers.deparcellate(node_coefs, self.parcellation_name)
        node_coefs_sig_surface = helpers.deparcellate(node_coefs_sig, self.parcellation_name)
        # save (and plot) surface maps
        if save:
            out_path = os.path.join(self.dir_path, f'correlation_{other.label.lower().replace(" ", "_")}_{test}')
            np.savez_compressed(
                out_path + '_nodewise_surface.npz', 
                coefs=node_coefs_surface, 
                coefs_sig = node_coefs_sig_surface,
                )
            pd.DataFrame({
                'coefs': node_coefs,
                'pvals': node_pvals,
                'pvals_fdr': node_pvals_fdr
            }).to_csv(
                out_path + '_nodewise_parcels.csv',
                index_label='parcel',
            )
        if plot:
            if plot_sig:
                surface = node_coefs_sig_surface
                filename = ((out_path + '_nodewise_surface_sig') if save else None)
            else:
                surface = node_coefs_surface
                filename = ((out_path + '_nodewise_surface_all') if save else None)
            vmin = min(
                np.nanmin(surface),
                -np.nanmax(surface)
            )
            helpers.plot_surface(
                surface, filename,
                cmap ='vlag', 
                vrange = (vmin , -vmin),
                cbar = True,
                layout_style = plot_layout
            )
            
    def associate_categorical_surface(self, categorical_surface, stats=True, 
            null_method='spin', n_perm=1000, save=False):
        """
        Calculates within and between categorical surface of the matrix 
        values and plots a collapsed matrix by its categories. 

        Parameters
        ----------
        categorical_surface: (surfaces.CategoricalSurface)
        stats: (bool)
            whether to do statistical test in addition to the plotting
        null_method: (str)
            - spin: rotates parcels
            - shuffle: uses np.shuffle for creating surrogates
        n_perm: (int)
            only used when stats=True
        save: (bool)
            save results as text and figure

        Returns
        ---------
        matrix_file: (str) path to matrix file    
        """
        # load cortical types and make it match matrix index
        parcellated_categories = categorical_surface.parcellated_data.iloc[:, 0]
        shared_parcels = self.matrix.index.intersection(parcellated_categories.index)
        matrix = self.matrix.loc[shared_parcels, shared_parcels]
        parcellated_categories = parcellated_categories.loc[shared_parcels].cat.remove_unused_categories()
        categories = parcellated_categories.cat.categories.tolist()
        n_categories = len(categories)
        # collapse matrix to mean values per cortical type
        mean_matrix_by_cortical_type = pd.DataFrame(np.zeros((n_categories, n_categories)),
                                                    columns=categories,
                                                    index=categories)
        for group_idx, group_df in matrix.groupby(parcellated_categories):
            mean_matrix_by_cortical_type.loc[group_idx, :] = \
                (group_df
                .T.groupby(parcellated_categories) # group columns by cortical type
                .mean() # take average of cortical types for each row (parcel)
                .mean(axis=1)) # take the average of average of rows in each cortical type
        # plot it
        helpers.plot_matrix(
            mean_matrix_by_cortical_type.values,
            self.file_path + f'_averaged-{categorical_surface.short_label}',
            vrange='sym', cmap=self.cmap
            )
        if stats:
            # quantify intra and intertype similarity
            intra_intertype = pd.DataFrame(
                np.zeros((n_categories+1, 2)),
                columns=['intra', 'inter'],
                index=categories+['All']
                )
            for group_idx, group_df in matrix.groupby(parcellated_categories):
                intra_intertype.loc[group_idx, 'intra'] = \
                    (group_df
                    .T.groupby(parcellated_categories) # group columns by cortical type
                    .mean() # take average of cortical types for each row (parcel)
                    .loc[group_idx].mean()) # take the average of average of rows in each cortical type
                intra_intertype.loc[group_idx, 'inter'] = \
                    (group_df
                    .T.groupby(parcellated_categories) # group columns by cortical type
                    .mean() # take average of cortical types for each row (parcel)
                    .drop(index=group_idx) # remove the same type
                    .values.mean()) # take the average of average of rows in each cortical type
            # calculate intra and inter type similarity across all types
            ## create a matrix indicating edges belonging to the same
            ## cortical type
            same_type_mat = (
                parcellated_categories.to_numpy()[:, np.newaxis] \
                == parcellated_categories.to_numpy()[np.newaxis, :]
                )
            ## average matrix values across edges with the same or different types
            intra_intertype.loc['All', 'intra'] = np.nanmean(matrix.to_numpy()[same_type_mat])
            intra_intertype.loc['All', 'inter'] = np.nanmean(matrix.to_numpy()[~same_type_mat])
            # test significance using permutation
            null_dist_intra_intertype = np.zeros((n_perm, n_categories+1, 2))
            surrogates = self.get_surrogates(null_method, n_perm)
            print(f"Calculating p-value with permutation testing ({n_perm} permutations)")
            for perm_idx in range(n_perm):
                if perm_idx % 100 == 0:
                    print("Perm", perm_idx)
                surrogate = surrogates[perm_idx].matrix.loc[matrix.index, matrix.columns]
                null_intra_intertype = pd.DataFrame(
                    np.zeros((n_categories+1, 2)),
                    columns=['intra', 'inter'],
                    index=categories+['All'])
                for group_idx, group_df in surrogate.groupby(parcellated_categories):
                    null_intra_intertype.loc[group_idx, 'intra'] = \
                        (group_df
                        .T.groupby(parcellated_categories) # group columns by cortical type
                        .mean() # take average of cortical types for each row (parcel)
                        .loc[group_idx].mean()) # take the average of average of rows in each cortical type
                    null_intra_intertype.loc[group_idx, 'inter'] = \
                        (group_df
                        .T.groupby(parcellated_categories) # group columns by cortical type
                        .mean() # take average of cortical types for each row (parcel)
                        .drop(index=group_idx) # remove the same type
                        .dropna() # drop unknown and ALO
                        .values.mean()) # take the average of average of rows in each cortical type
                null_intra_intertype.loc['All', 'intra'] = np.nanmean(surrogate.to_numpy()[same_type_mat])
                null_intra_intertype.loc['All', 'inter'] = np.nanmean(surrogate.to_numpy()[~same_type_mat])
                null_dist_intra_intertype[perm_idx, :, :] = null_intra_intertype.values
            null_dist_diff_intra_inter = null_dist_intra_intertype[:, :, 0] - null_dist_intra_intertype[:, :, 1]
            diff_intra_inter = (intra_intertype.iloc[:, 0] - intra_intertype.iloc[:, 1]).values.reshape(1, -1)
            intra_intertype['pvals'] = (null_dist_diff_intra_inter > diff_intra_inter).mean(axis=0)
            if save:
                intra_intertype.to_csv(self.file_path + f'_{categorical_surface.short_label}_intra_inter_diff.txt')
            return intra_intertype

    def get_surrogates(self, method='spin', n_perm=1000):
        """
        Create surrogate matrices by rotating parcels

        Parameters
        ---------
        method: (str)
            - spin
            - shuffle
        n_perm: (int)

        Returns
        -------
        surrogates: (list of Matrix)
        """
        if method == 'spin':
            # get the list of parcels excluded from the matrix
            all_parcels = helpers.parcellate(
                helpers.deparcellate(self.matrix.iloc[:,0], self.parcellation_name, downsampled=True), 
                self.parcellation_name
            ).drop(index=helpers.MIDLINE_PARCELS.get(self.parcellation_name, [])).index
            excluded_parcels = all_parcels.difference(self.matrix.index).tolist()
            # get rotated parcel surrogates
            surrogate_parcel_orders = helpers.get_rotated_parcels(self.parcellation_name, n_perm, 
                excluded_parcels=excluded_parcels, return_indices=False)
        elif method == 'shuffle':
            surrogate_parcel_orders = []
            for perm_idx in range(n_perm):
                surrogate_parcel_orders.append(
                    np.random.permutation(self.matrix.index.tolist())[:, np.newaxis]
                )
            surrogate_parcel_orders = np.concatenate(surrogate_parcel_orders, axis=1)
            print(surrogate_parcel_orders.shape)
        # apply surrogate parcel orders
        surrogates = []
        for i in range(n_perm):
            surrogate_matrix = self.matrix.copy()
            surrogate_matrix.index = surrogate_matrix.columns = surrogate_parcel_orders[:, i]
            surrogate_matrix = surrogate_matrix.loc[self.matrix.index, self.matrix.columns]
            surrogate = Matrix(surrogate_matrix, self.parcellation_name, 
                self.label+f'_surrogate{i}')
            surrogates.append(surrogate)
        return surrogates
    
    def binned_average(self, surf, column, nbins=10):
        """
        Returns the matrix averaged per bins of surf[column]
        """
        bins = pd.qcut(surf.parcellated_data.loc[:, column], 10).cat.codes
        shared_parcels = bins.index.intersection(self.matrix.index)
        bins = bins.loc[shared_parcels]
        matrix = self.matrix.loc[shared_parcels, shared_parcels]
        matrix_collapsed = np.zeros((nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                matrix_collapsed[i, j] = \
                    np.nanmean(matrix.loc[bins[bins==i].index, bins[bins==j].index].values)
        return matrix_collapsed

class CurvatureSimilarityMatrix(Matrix):
    """
    Matrix showing similarity of curvature distribution
    between each pair of parcels
    """
    def __init__(self, parcellation_name, exc_regions=None):
        """
        Creates curvature similarity matrix or loads it if it already exist

        Parameters
        ----------
        parcellation_name: (str)
        exc_regions: (str | None)
        """
        self.parcellation_name = parcellation_name
        self.exc_regions = exc_regions
        self.label = "Curvature similarity"
        self.cmap = sns.color_palette("mako", as_cmap=True)
        self.dir_path = os.path.join(OUTPUT_DIR, 'curvature')
        self.file_path = os.path.join(
            self.dir_path,
            f'curvature_similarity_matrix_parc-{parcellation_name}'
            )
        if os.path.exists(self.file_path + '.npz'):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self.matrix = self._create()
            self._save()
        # remove the adysgranular parcels after the matrix is created
        # to avoid unncessary waste of computational resources
        # note this is not saved
        if self.exc_regions:
            self.matrix = self._remove_parcels(self.exc_regions)
        
    def _create(self):
        """
        Create curvature similarity matrix by calculating Jansen-Shannon similarity of
        curvature distributions between pairs of parcels        
        """
        # load curvature maps
        curvature_maps = datasets.load_curvature_maps()
        # parcellate curvature
        parcellated_curvature = helpers.parcellate(
            curvature_maps, 
            self.parcellation_name, 
            averaging_method=None,
            na_midline=False,
            )
        # create pdfs and store min and max curv at each parcel
        pdfs = {}
        for hem in ['L', 'R']:
            pdfs[hem] = parcellated_curvature[hem].apply(
                lambda group_df: pd.Series({
                    'pdf': scipy.stats.gaussian_kde(group_df[0]),
                    'min_curv': group_df[0].min(),
                    'max_curv': group_df[0].max(),
                })
                )
        # concatenate L and R hemisphere parcels,
        # dropping parcels duplicated in both hemispheres
        pdfs = (pd.concat([pdfs['L'], pdfs['R']], axis=0)
                .reset_index(drop=False)
                .drop_duplicates('index')
                .set_index('index')
        )
        # measure parcel-to-parcel similarity of curvature distributions
        # using Jensen-Shannon distance
        js_distance_matrix = pd.DataFrame(
            np.zeros((pdfs.shape[0],pdfs.shape[0])),
            columns=pdfs.index,
            index=pdfs.index
        )
        print("Calculating curvature similarity. This may take a while!")
        for parc_i, pdf_i in pdfs.iterrows():
            for parc_j, pdf_j in pdfs.iterrows():
                if parc_i == parc_j:
                    js_distance_matrix.loc[parc_i, parc_j] = 0
                elif parc_i > parc_j: # lower triangle only
                    # find the min and max curv across the pair of parcels and 
                    # create a linearly spaced discrete array [min, max]
                    # used for sampling PDFs of curvature in each parcel
                    pair_min_curv = min(pdf_i['min_curv'], pdf_j['min_curv'])
                    pair_max_curv = min(pdf_i['max_curv'], pdf_j['max_curv'])
                    X_pair = np.linspace(pair_min_curv, pair_max_curv, 200)
                    # sample PDFi and PDFj at X_pair and convert it to discrete
                    # PDF via dividing by sum
                    Y_i = pdf_i['pdf'].evaluate(X_pair)
                    Y_j = pdf_j['pdf'].evaluate(X_pair)
                    Y_i /= Y_i.sum()
                    Y_j /= Y_j.sum()
                    js_distance_matrix.loc[parc_i, parc_j] = scipy.spatial.distance.jensenshannon(Y_i, Y_j)    # calcualte curvature similarity as 1 - distance (TODO: is this the best approach?)
        # make sure that there are no np.infs and the distance is bound by 0 and 1
        assert (js_distance_matrix.values.min() >= 0) and (js_distance_matrix.values.max() <= 1)
        # calcualate similarity as 1 - dintance
        curv_similarity_matrix = 1 - js_distance_matrix
        # copy the lower triangle to the upper triangle
        i_upper = np.triu_indices(curv_similarity_matrix.shape[0], 1)
        curv_similarity_matrix.values[i_upper] = curv_similarity_matrix.T.values[i_upper]
        return curv_similarity_matrix

class DistanceMatrix(Matrix):
    def __init__(self, parcellation_name, kind='geodesic', 
            approach='center-to-center', exc_regions=None):
        """
        Matrix of geodesic/Euclidean distance between centroids of parcels

        Parameters
        ----------
        parcellation_name: (str) name of the parcellation (must be stored in data/parcellations)
        kind: (str)
            - geodesic (Default)
            - euclidean
        approach: (str) only used for geodesic distance
            - center-to-center: calculate pair-wise distance between centroids of parcels. Results in symmetric matrix.
            - center-to-parcel: calculates distance between centroid of one parcel to all vertices
                                in the other parcel, taking the mean distance. Can result in asymmetric matrix.
                                (this is "geoDistMapper.py" behavior)
        exc_regions: (str | None)

        Based on "geoDistMapper.py" from micapipe/functions
        Original Credit:
        # Translated from matlab:
        # Original script by Boris Bernhardt and modified by Casey Paquola
        # Translated to python by Jessica Royer
        """
        self.parcellation_name = parcellation_name
        if self.parcellation_name == 'M132':
            self.space = 'yerkes'
        else:
            self.space = 'bigbrain'
        self.kind = kind
        self.approach = approach
        self.exc_regions = exc_regions
        self.label = f"{self.kind.title()} distance"
        self.split_hem = True
        self.cmap = sns.color_palette("viridis", as_cmap=True)
        self.dir_path = os.path.join(OUTPUT_DIR, 'distance')
        self.file_path = os.path.join(
            self.dir_path,
            f'{self.kind}_distance_matrix_parc-{self.parcellation_name}_approach-{self.approach}'
            )
        if os.path.exists(self.file_path + '.npz'):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            # create the matrix
            if self.kind == 'geodesic':
                self.matrix = self._create_geodesic()
            else:
                self.matrix = self._create_euclidean()
            # remove midline parcels
            midline_parcels = helpers.MIDLINE_PARCELS.get(self.parcellation_name, [])
            self.matrix = self.matrix.drop(
                index=midline_parcels, 
                columns=midline_parcels, 
                errors='ignore')
            self._save()
            self.plot(vrange=(0, 1))
        # remove the adysgranular parcels after the matrix is created
        # to avoid unncessary waste of computational resources
        # note this is not saved or plotted
        if self.exc_regions:
            self.matrix = self._remove_parcels(self.exc_regions)

    def _create_geodesic(self):
        """
        Creates center-to-parcel or center-to-center geodesic distance matrix
        between pairs  of parcels
        """
        parcel_centers = helpers.get_parcel_center_indices(
            self.parcellation_name, kind='orig', space=self.space, downsampled=False)
        GDs = {}
        for hem in ['L', 'R']:
            # load surf
            if self.space == 'bigbrain':
                surf_path = os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                )
            else:
                surf_path = os.path.join(
                    SRC_DIR, f'MacaqueYerkes19.{hem}.pial.32k_fs_LR.surf.gii'
                )
            parc = datasets.load_parcellation_map(self.parcellation_name, False, space=self.space)[hem]
            GDs[hem] = pd.DataFrame()
            print(f"\nRunning geodesic distance in hemisphere {hem}\nParcel:", end=" ")
            for i, (parcel, center_vertex) in enumerate(parcel_centers[hem].items()):
                print(i, end=" ")
                cmdStr = f"{wbPath} -surface-geodesic-distance {surf_path} {center_vertex} {self.file_path}_this_voi.func.gii"
                subprocess.run(cmdStr.split())
                tmpname = self.file_path + '_this_voi.func.gii'
                tmp = nibabel.load(tmpname).agg_data()
                os.remove(tmpname)
                for other_parcel, other_center in parcel_centers[hem].items():
                    if self.approach=='center-to-parcel':
                        tmpData = tmp[parc == other_parcel]
                        GDs[hem].loc[parcel, other_parcel] = np.mean(tmpData)
                    elif self.approach=='center-to-center':
                        GDs[hem].loc[parcel, other_parcel] = tmp[other_center]
        # join the GD matrices from left and right hemispheres
        GD_full = (pd.concat([GDs['L'], GDs['R']],axis=0)
                .reset_index(drop=False)
                .drop_duplicates('index')
                .set_index('index'))
        return GD_full

    def _create_euclidean(self):
        """
        Creates center-to-center euclidean distance matrix
        between pairs  of parcels
        """
        center_indices = helpers.get_parcel_center_indices(
            self.parcellation_name, kind='inflate', space=self.space, downsampled=False)
        center_coords = {}
        for hem in ['L', 'R']:
            if self.space == 'bigbrain':
                surf_path = os.path.join(
                    SRC_DIR, f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                )
            else:
                surf_path = os.path.join(
                    SRC_DIR, f'MacaqueYerkes19.{hem}.pial.32k_fs_LR.surf.gii'
                )
            coords = nilearn.surface.load_surf_mesh(surf_path).coordinates
            center_coords[hem] = coords[center_indices[hem].values]
        center_coords = np.vstack([center_coords['L'], center_coords['R']])
        ED_matrix = scipy.spatial.distance_matrix(center_coords, center_coords)
        parcels = center_indices['L'].index.to_list() + center_indices['R'].index.to_list()
        ED_matrix = pd.DataFrame(ED_matrix, columns=parcels, index=parcels)
        return ED_matrix

    def regress_out(self, other, plot=True, save_plot=False, stats_on_plot=True, 
            spin_test=False, return_r2=False, n_perm=1000):
        """
        Regresses out GD matrix from another matrix (e.g. LTC)
        using a exponential fit
        """
        # make sure they have the same parcellation and mask
        assert self.parcellation_name == other.parcellation_name
        # match the matrices in the order and selection of parcels
        # + convert them to np.ndarray
        shared_parcels = self.matrix.index.intersection(other.matrix.index)
        GD = self.matrix.loc[shared_parcels, shared_parcels]
        Y = other.matrix.loc[shared_parcels, shared_parcels]
        # get the index for lower triangle
        tril_index = np.tril_indices_from(GD.values, -1)
        gd = GD.values[tril_index]
        y = Y.values[tril_index]
        # remove NaNs (e.g. interhemispheric pairs)
        mask = ~(np.isnan(gd) | np.isnan(y))
        # keep track of which items are in the mask which will be needed
        # when reconstructing unmasked full matrix
        mask_idx = np.arange(mask.shape[0])[mask]
        gd = gd[mask]
        y = y[mask]
        # the exponential fit
        coefs = helpers.exponential_fit(gd, y)
        # get the reisd
        y_hat = helpers.exponential_eval(gd, *coefs)
        y_resid = y - y_hat
        # calculate R2
        r2 = 1 - (y_resid.var() / y.var())
        if spin_test:
            surrogate_matrices = other.get_surrogates(method='spin', n_perm=n_perm)
            null_dist = np.zeros(n_perm)
            for i, surrogate in enumerate(surrogate_matrices):
                null_dist[i] = self.regress_out(surrogate, plot=False, spin_test=False, return_r2=True)
            p_val = (np.abs(null_dist) >= np.abs(r2)).mean()
        if plot:
            # plot the polynomial fit
            sns.set_style('ticks')
            fig, ax = plt.subplots(figsize=(6, 4), dpi=192)
            ax = sns.scatterplot(gd, y, s=1, alpha=0.2, color='grey')
            line_x = np.linspace(0, gd.max(), 500)
            line_y = helpers.exponential_eval(line_x, *coefs)
            ax.plot(line_x, line_y, color='red')
            ax.set_xlabel('Geodesic distance')
            ax.set_ylabel(other.label)
            if stats_on_plot:
                if spin_test:
                    text = f'R2 = {r2:.2f}; p = {p_val:.2f}'
                else:
                    text = f'R2 = {r2:.2f}'
                ax.text(
                    ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.60,
                    ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.90,
                    text,
                    color='black', size=12,
                    multialignment='left')
            if save_plot:
                fig.tight_layout()
                fig.savefig(other.file_path+'_gd_regression')
        if return_r2:
            if spin_test:
                return r2, p_val
            else:
                return r2
        else:
            # convert it back to the matrix
            lhalf_matrix = np.zeros_like(tril_index[0]).astype('float64')
            lhalf_matrix[mask_idx] = y_resid
            Y_RES = np.zeros_like(Y)
            Y_RES[tril_index] = lhalf_matrix
            Y_RES = Y_RES.T
            Y_RES[tril_index] = lhalf_matrix
            Y_RES[np.diag_indices_from(Y_RES)] = np.diag(Y)
            # label the matrix
            Y_RES = pd.DataFrame(
                Y_RES, index=shared_parcels, columns=shared_parcels
            )
            # set interhemispheric edges to NaN
            hem_parcels = helpers.get_hem_parcels(
                self.parcellation_name, 
                limit_to_parcels=shared_parcels.tolist()
                )
            Y_RES.loc[hem_parcels['R'], hem_parcels['L']] = np.NaN
            Y_RES.loc[hem_parcels['L'], hem_parcels['R']] = np.NaN
            return Y_RES

class ConnectivityMatrix(Matrix):
    def __init__(self, kind, exc_regions=None, exc_contra=True, 
            parcellation_name='schaefer400', dataset='hcp',
            threshold=False, long_range=False):
        """
        Structural, functional or effective connectivity matrix

        Parameters
        ---------
        kind: (str)
            - structural
            - functional
            - effective
        exc_regions: (str | None)
        sc_zero_contra: (bool) zero out contralateral edges for the SC matrix
        parcellation_name: (str)
            - schaefer400 (currently only this is supported)
        dataset: (str)
            - hcp
            - mics (only for EC)
        threshold: (bool)
            keep only the values of connected edges
        create_plot: (bool)
        """
        # TODO: create separate classes for SC/FC and EC
        self.kind = kind
        self.exc_regions = exc_regions
        self.exc_contra = exc_contra
        self.threshold = threshold
        self.long_range = long_range
        if self.kind == 'structural':
            self.cmap = 'bone'
        if self.exc_contra:
            self.split_hem = True
        elif self.kind == 'functional':
            self.cmap = cmcrameri.cm.davos
        else:
            self.cmap = 'rocket'
        self.parcellation_name = parcellation_name
        self.label = f'{self.kind.title()} connectivity'
        self.dataset = dataset
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'connectivity',
            f'{self.kind[0]}c'\
            + f'_parc-{self.parcellation_name}' \
            + ('_split_hem' if self.split_hem else '')
            + f'_data-{self.dataset}'
            )
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        # load the matrix
        self.matrix = datasets.fetch_conn_matrix(self.kind, self.parcellation_name, self.dataset)
        # threshold it if indicated
        if self.threshold:
            if self.kind == 'functional':
                connected_mask = self.binarize().values.astype('bool')
                self.matrix.iloc[~(connected_mask)] = np.NaN
            elif self.kind == 'structural':
                self.matrix.values[np.isclose(self.matrix, 0)] = np.NaN
        if self.exc_contra:
            hem_parcels = helpers.get_hem_parcels(
                self.parcellation_name, 
                limit_to_parcels=self.matrix.index.tolist()
                )
            self.matrix.loc[hem_parcels['L'], hem_parcels['R']] = np.NaN
            self.matrix.loc[hem_parcels['R'], hem_parcels['L']] = np.NaN
        # remove the adysgranular parcels after the matrix is created
        # to avoid unncessary waste of computational resources
        # note this is not saved
        if self.exc_regions:
            self.matrix = self._remove_parcels(self.exc_regions)
        if self.long_range:
            gd = DistanceMatrix(self.parcellation_name)
            shared_parcels = self.matrix.index.intersection(gd.matrix.index)
            gd = gd.matrix.loc[shared_parcels, shared_parcels]
            self.matrix = self.matrix.loc[shared_parcels, shared_parcels]
            self.matrix.values[(gd < 75).values] = np.NaN

    def binarize(self, fc_pthreshold=0.2, plot=False):
        """
        Binarize the SC or FC
        """
        # keep track of cells that are NaN
        nan_mask = self.matrix.isna()
        if self.kind == 'structural':
            mat_binned = pd.DataFrame(
                        self.matrix.values > 0,
                        index = self.matrix.index,
                        columns = self.matrix.columns
                    )
        elif self.kind == 'functional':
            mat_binned = self.matrix >= np.nanquantile(self.matrix.values.flatten(), 1-fc_pthreshold)
        else:
            raise NotImplementedError
        mat_binned = mat_binned.mask(nan_mask, np.NaN)
        if plot:
            helpers.plot_matrix(
                mat_binned.values.astype('float'),
                self.file_path + '_bin',
                cmap = 'binary',
                vrange=(0, 1),
            )
        return mat_binned

    def binarized_association(self, others, fc_pthreshold=0.2, 
            spin_test=False, n_perm=1000, verbose=True, 
            plot=True, stats_on_plot=True):
        """
        Binarize the SC or FC matrix and performs logistic regression
        with FC/SC as DV and `others` as IV. If `other` is categorical
        (e.g. CorticalTypeDiffMatrix) it also plots a stacked bar plot
        of existing vs absent connections per category

        Parameters
        ---------


        Returns
        -------
        lgrs: (list of statsmodels.discrete.discrete_model.Logit)
        pvals: (list of float)
            only if spin_test=True
        null_lgrs: (list of list of statsmodels.discrete.discrete_model.Logit)
            only if spin_test=True
        """
        if not isinstance(others, list):
            others = [others]
        # binarize connectivity
        Y = self.binarize(fc_pthreshold=fc_pthreshold).astype('float')
        # get the shared parcels across self and all others
        shared_parcels = Y.index
        for other in others:
            shared_parcels = shared_parcels.intersection(other.matrix.index)
        # get shared parcels of Y and convert its lower triangle into 1d array
        Y = Y.loc[shared_parcels, shared_parcels]
        tril_index = np.tril_indices_from(Y.values, -1)
        y = Y.values[tril_index]
        # create a mask for removing NaNs (e.g. interhemispheric pairs)
        ## converting array to float because
        ## np.isnan(boolean_array_with_nans) throws an error
        mask = ~(np.isnan(y))
        x_arrays = {}
        x_dtypes = {}
        for other in others:
            # get shared parcels of X and its lower triangle
            X = other.matrix.loc[shared_parcels, shared_parcels]
            x = X.values[tril_index]
            x_name = other.label.lower().replace(' ', '_')
            x_arrays[x_name] = x
            x_dtypes[x_name] = other.matrix.iloc[:, 0].dtype.name
            # remove NaN values of x from the mask
            mask = mask & ~(np.isnan(x.astype('float')))
        # apply the mask
        y = y[mask].astype('bool')
        for x_name, x in x_arrays.items():
            x_arrays[x_name] = x[mask]
        # create a dataframe
        df = pd.DataFrame(x_arrays)
        df['Connected'] = y
        # logistic regression
        # if others is > 1 this will perform a nested
        # logistic regression adding each matrix to the
        # model one by one, and tests fit improvement with
        # likelihood ratio test
        lgrs = []
        for x_idx in range(len(others)):
            lgr = sm.Logit(
                df['Connected'], # dependent variable 
                sm.add_constant(df.iloc[:, :x_idx+1]) # independent variable(s)
                ).fit(disp=verbose)
            if verbose:
                # print the results
                res_str = str(lgr.summary())
                print(res_str)
                # print the ORs
                print(
                    pd.DataFrame({
                        'OR': np.exp(lgr.params),
                        'L_CI': np.exp(lgr.params - (lgr.bse * 1.96)),
                        'U_CI': np.exp(lgr.params + (lgr.bse * 1.96)),
                    })
                )
                # perform likelihood ratio test after the
                # second model with its previous model
                if x_idx > 0:
                    likelihood_ratio = -2*(lgrs[-1].llf - lgr.llf)
                    # do chi-squared test with df = 1 (as only one
                    # IV is added with respect to the previous model)
                    p_val = scipy.stats.chi2.sf(likelihood_ratio, 1)
                    print(f"\n\nLikelihood ratio: {likelihood_ratio}, p-value: {p_val}\n\n")
            lgrs.append(lgr)
        if plot:
            for x_name in x_arrays.keys():
                # for continous x_arrays plot the probability of connection 
                # per non-overlapping window of x value
                if x_dtypes[x_name] != 'category':
                    ## sort the edges by x values
                    sorted_df = df.sort_values(x_name).reset_index()
                    ## segment the df into 200 windows and take
                    ## the mean of x and y per segment
                    n_windows = 200
                    segmented_df = sorted_df.groupby(sorted_df.index // n_windows).mean()
                    ## create a line plot
                    fig, ax = plt.subplots(dpi=192)
                    ax.plot(segmented_df[x_name], segmented_df['Connected'], color='grey', marker='.', ls='')
                    ax.set_ylabel('Prob. Connected')
                # for categorical x_arrays plot the stacked bar existing vs absent 
                # connections grouped by categories
                else:
                    ## calculate the percentage of existing and absent connections per category
                    percentages = (
                        df.groupby(x_name)['Connected']
                        .value_counts(normalize=True)
                        .mul(100).rename('percentage')
                        .reset_index()
                    )
                    ## add categories with no existing or absent connections
                    ## to the percentages to prevent errors when creating the barplots
                    for single_bin_cat in percentages.value_counts(x_name)[percentages.value_counts(x_name) == 1].keys():
                        bin_w_100_percent = percentages.loc[percentages[x_name]==single_bin_cat, 'Connected'].values[0]
                        bin_w_0_percent = ~bin_w_100_percent
                        percentages = percentages.append({x_name: single_bin_cat, 'Connected': bin_w_0_percent, 'percentage':0.0}, ignore_index=True)
                    print(percentages)
                    percentages_all = (df['Connected']
                        .value_counts(normalize=True)
                        .mul(100).rename('percentage')
                        .reset_index())
                    print(percentages_all)
                    ## create the bar plots
                    fig, ax = plt.subplots(dpi=192)
                    existing_conn = percentages[percentages['Connected']]
                    absent_conn = percentages[~percentages['Connected']]
                    ax.bar(existing_conn[x_name], height=existing_conn['percentage'], facecolor='black', edgecolor='black', width = 0.5)
                    # ax.bar(absent_conn[x_name], bottom=existing_conn['percentage'], height=absent_conn['percentage'], facecolor='white', edgecolor='black', width = 0.5)
                    ax.set_xticks(percentages[x_name].unique())
                    ax.set_ylabel('Connected %')
                if (len(others) == 1) & stats_on_plot:
                    if lgrs[0].params[x_name] >= 0:
                        text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
                        text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.90
                    else:
                        text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.75
                        text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.90
                    ax.text(text_x, text_y, f'R2 = {lgrs[0].prsquared:.3f}',
                            color='black', size=16,
                            multialignment='left')
                ax.set_xlabel(f'{x_name.replace("_"," ").title()}')
        if spin_test:
            others_surrogates = []
            for other in others:
                others_surrogates.append(other.get_surrogates('spin', n_perm))
            null_lgrs = []
            null_dist_r2 = [np.zeros(n_perm) for _ in range(len(lgrs))]
            for i in range(n_perm):
                surrogate_others = []
                for j in range(len(others)):
                    surrogate_others.append(others_surrogates[j][i])
                null_lgrs.append(self.binarized_association(surrogate_others, fc_pthreshold,
                    spin_test=False, verbose=False, plot=False))
                for lgr_i, null_lgr in enumerate(null_lgrs[-1]):
                    null_dist_r2[lgr_i][i] = null_lgr.prsquared
            pvals = []
            for lgr_i in range(len(lgrs)):
                pvals.append((null_dist_r2[lgr_i] >= lgrs[lgr_i].prsquared).mean())
            return lgrs, pvals, null_lgrs
        return lgrs

class MicrostructuralCovarianceMatrix(Matrix):
    """
    Matrix showing microstructural similarity of parcels in their relative laminar
    thickness, relative laminar volume, density profiles (MPC), or their combination
    """
    def __init__(self, input_type, parcellation_name='sjh', 
        exc_regions='adysgranular', correct_curvature='smooth-10', 
        regress_out_geodesic_distance = False, relative = True,
        similarity_metric='parcor', merge_layers=None,
        zero_out_negative=False, laminar_density=False):
        """
        Initializes laminar similarity matrix object

        Parameters
        ---------
        input_type: (str)
            - 'thickness' [default]: laminar thickness from BigBrain
            - 'density': profile density from BigBrain
            - 'thickness-density': fused laminar thickness and profile density from BigBrain
        parcellation_name: (str | None) (only for bigbrain)
        exc_regions: (str | None)
        correct_curvature: (str or None) ignored for 'density'
            - 'smooth-{radius}' [Default]: smooth the thickness using a disc on the inflated surface
            - 'volume': normalize relative thickness by curvature according to the 
            equivolumetric principle i.e., use relative laminar volume instead of relative laminar 
            thickness. Laminar volume is expected to be less affected by curvature.
            - 'regress': regresses map of curvature out from the map of relative thickness
            of each layer (separately) using a linear regression. This is a more lenient approach of 
            controlling for curvature.
            - None
        regress_out_geodesic_distance: (bool)
            uses a polynomial of degree 2 to regress out geodesic distance from the matrix
            Note that this will make interhemispheric edges NaN
        similarity_metric: (str) how is similarity of laminar structure between two parcels determined
            - 'parcor': partial correlation with mean thickness pattern as covariate
            - 'euclidean': euclidean distance inverted and normalize to 0-1
            - 'pearson': Pearson's correlation coefficient
        merge_layers: (None | list of list)
            if provided it should be a nested list of the layer indices (0-5) 
            that should be merged
        zero_out_negative: (bool)
            zero out negative values from the matrix
        laminar_density: (bool)
            when input is density load laminar instead of total depth density
        """
        # save parameters as class fields
        self.input_type = input_type
        self.correct_curvature = correct_curvature
        self.regress_out_geodesic_distance = regress_out_geodesic_distance
        if self.regress_out_geodesic_distance:
            self.split_hem = True
        self.similarity_metric = similarity_metric
        self.parcellation_name = parcellation_name
        self.exc_regions = exc_regions
        self.zero_out_negative = zero_out_negative
        self.relative = relative
        self.laminar_density = laminar_density
        self.merge_layers = merge_layers
        # set the label based on input type
        SHORT_LABELS = {
            'thickness': 'LTC',
            'density': 'MPC',
            'thickness-density': 'Fused'
        }
        LABELS = {
            'thickness': 'Laminar thickness covariance',
            'density': 'Microstructural profile covariance',
            'thickness-density': 'Laminar thickness and microstructural profile covariance'
        }
        self.label = LABELS[self.input_type]
        if self.regress_out_geodesic_distance:
            self.label += ' (regressed out GD)'
        self.short_label = SHORT_LABELS[self.input_type]
        # set cmap
        if self.input_type == 'density':
            self.cmap = sns.color_palette("rocket", as_cmap=True)
        else: # TODO: use different colors for each input type in the plot in thickness-density
            self.cmap = sns.color_palette("YlGnBu_r", as_cmap=True)
        # directory and filename (prefix which will be used for .npz and .jpg files)
        self.dir_path = self._get_dir_path()
        os.makedirs(self.dir_path, exist_ok=True)
        self.file_path = os.path.join(self.dir_path, 'matrix')
        if os.path.exists(self.file_path + '.npz'):
            if self.parcellation_name is not None:
                # for the unparcellated data we really don't care about
                # the matrix and only need the matrix object for locating
                # the gradients, and therefore there's no need to load
                # the actual huge matrix!
                self._load()
        else:
            self._create()
            self._save()

    def _create(self):
        """
        Creates microstructural covariance matrix
        """
        print("Creating microstructural covariance matrix")
        if self.input_type == 'thickness-density':
            self.children = {}
            for input_type in ['thickness', 'density']:
                # create matrix_obj for each input_type
                # by calling the same class
                matrix_obj = MicrostructuralCovarianceMatrix(
                    input_type=input_type,
                    parcellation_name=self.parcellation_name,
                    exc_regions=self.exc_regions,
                    correct_curvature=self.correct_curvature,
                    regress_out_geodesic_distance=self.regress_out_geodesic_distance,
                    similarity_metric=self.similarity_metric,
                    zero_out_negative=self.zero_out_negative,
                    laminar_density=self.laminar_density
                )
                self.children[input_type] = matrix_obj
                #TODO: it is unlikely but maybe valid parcels are different for each modality
                # in that case this code won't work properly
            self.matrix = self._fuse_matrices([child.matrix for child in self.children.values()])
        else:
            # Load laminar thickness or density profiles
            self._load_input_data()
            # create the similarity matrix
            self.matrix = self._calculate_similarity(self._parcellated_input_data)
            if self.regress_out_geodesic_distance:
                GD = DistanceMatrix(
                    self.parcellation_name,
                    kind = 'geodesic',
                    exc_regions = self.exc_regions
                    )
                self.matrix = GD.regress_out(self)

    def _load_input_data(self):
        """
        Load input data
        """
        # load the data
        if self.input_type == 'thickness':
            if self.correct_curvature is None:
                self._input_data = datasets.load_laminar_thickness(
                    exc_regions=self.exc_regions,
                    regress_out_curvature=False,
                    normalize_by_total_thickness=self.relative,
                )
            elif 'smooth' in self.correct_curvature:
                smooth_disc_radius = int(self.correct_curvature.split('-')[1])
                self._input_data = datasets.load_laminar_thickness(
                    exc_regions=self.exc_regions,
                    regress_out_curvature=False,
                    normalize_by_total_thickness=self.relative,
                    smooth_disc_radius=smooth_disc_radius
                )
                # note that in this case the input data is in ico5 space
                # which may lead to some missing parcels in e.g. sjh
                # TODO: make sure this will not cause any problems
            elif self.correct_curvature == 'volume':
                # Note: later turned out that this does not 
                # correct for curvature properly, but I'm keeping
                # it for now for backward compatibility
                self._input_data = datasets.load_laminar_volume(
                    exc_regions=self.exc_regions,
                )
            elif self.correct_curvature == 'regress':
                self._input_data = datasets.load_laminar_thickness(
                    exc_regions=self.exc_regions,
                    regress_out_curvature=True,
                    normalize_by_total_thickness=self.relative,
                )
        elif self.input_type == 'density':
            if self.laminar_density:
                self._input_data = datasets.load_laminar_density(
                    exc_regions=self.exc_regions
                )
            else:
                self._input_data = datasets.load_total_depth_density(
                    exc_regions=self.exc_regions
                )
        # downsample the data if it's not already downsampled
        # (for homogeneity of different types of matrices)
        if self._input_data['L'].shape[0] == datasets.N_VERTICES_HEM_BB:
            self._input_data = helpers.downsample(self._input_data)
        # parcellate input data
        concat_input_data = np.concatenate([self._input_data['L'], self._input_data['R']], axis=0)
        if self.parcellation_name is not None:
            # concatenate and parcellate
            self._parcellated_input_data = helpers.parcellate(
                concat_input_data,
                self.parcellation_name,
                averaging_method='median'
                )
            # renormalize the parcellated relative laminar thickness
            if (self.input_type == 'thickness') & self.relative:
                self._parcellated_input_data = \
                    self._parcellated_input_data.divide(
                        self._parcellated_input_data.sum(axis=1), 
                        axis=0
                        )
        else:
            # for consistency of unparcellated matrix object
            # with parcellated ones convert input data to a
            # dataframe and call it _parcellated_input_data
            self._parcellated_input_data = pd.DataFrame(concat_input_data)
        # remove NaNs. If there are NaN values in the parcellated_input_data there will
        # be all-zero rows in the matrix and this will makes all the elements
        # in the gradient = NaN. There shouldn't usually be NaNs at this point
        # but still there is one parcel in schaefer1000. TODO: fix this in the source
        self._parcellated_input_data = self._parcellated_input_data.dropna()
        if self.parcellation_name is not None:
            # get only the valid parcels outside of exc_regions or midline
            # which also exist in self._parcellated_input_data
            self.valid_parcels = helpers.get_valid_parcels(
                self.parcellation_name, 
                self.exc_regions, 
                downsampled=True).intersection(self._parcellated_input_data.index)
        else:
            # get non-NA vertices
            self.valid_parcels = self._parcellated_input_data[
                ~(self._parcellated_input_data.isna().any(axis=1))
            ].index
        self._parcellated_input_data = self._parcellated_input_data.loc[self.valid_parcels]
        if self.merge_layers is not None:
            layer_group_dfs = []
            for layer_group in self.merge_layers:
                layer_group_dfs.append(self._parcellated_input_data.iloc[:, layer_group].sum(axis=1))
            self._parcellated_input_data = pd.concat(layer_group_dfs, axis=1)

    def _calculate_similarity(self, parcellated_input_data, transform=True):
        """
        Creates laminar similarity matrix by taking Euclidean distance, Pearson's correlation
        or partial correltation (with the average laminar data pattern as the covariate) between
        average values of parcels

        Note: Partial correlation is based on "Using recursive formula" subsection in the wikipedia
        entry for "Partial correlation", which is also the same as Formula 2 in Paquola et al. PBio 2019
        (https://doi.org/10.1371/journal.pbio.3000284)
        Note 2: Euclidean distance is reversed (* -1) and rescaled to 0-1 (with 1 showing max similarity)

        Parameter
        --------
        parcellated_input_data: (pd.DataFrame) n_valid_parcels x n_features
        transform: (bool)
            Whether to perfrom transformations on the raw matrix. Default: True.

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                microstructure (laminar thickness or density profiles)
        """
        # Calculate parcel-wise similarity matrix
        if self.similarity_metric in ['parcor', 'pearson']:
            if self.similarity_metric == 'parcor':
                r_ij = np.corrcoef(parcellated_input_data)
                mean_input_data = parcellated_input_data.mean()
                r_ic = np.corrcoef(
                    parcellated_input_data.values, 
                    mean_input_data.values[np.newaxis, :])[-1, :-1] # r_ic and r_jc are the same
                r_icjc = np.outer(r_ic, r_ic) # the second r_ic is actually r_jc
                matrix = (r_ij - r_icjc) / np.sqrt(np.outer((1-r_ic**2),(1-r_ic**2)))
            else:
                matrix = np.corrcoef(parcellated_input_data.values)
            if transform:
                # zero out negative correlations
                if self.zero_out_negative:
                    matrix[matrix<0] = 0
                # zero out correlations of 1 (to avoid division by 0)
                matrix[np.isclose(matrix, 1)] = 0
                # Fisher's z-transformation
                matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
                # zero out NaNs and inf
                matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        elif self.similarity_metric == 'euclidean':
            # calculate pair-wise euclidean distance
            matrix = sk.metrics.pairwise.euclidean_distances(parcellated_input_data.values)
            if transform:
                # make it negative (so higher = more similar) and rescale to range (0, 1)
                matrix = sk.preprocessing.minmax_scale(-matrix, (0, 1))
        # label the matrix
        matrix = pd.DataFrame(
            matrix, 
            index=self.valid_parcels,
            columns=self.valid_parcels)
        return matrix

    def _fuse_matrices(self, matrices):
        """
        Fuses two input matrices by rank normalizing each and rescaling
        the second matrix based on the first one

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
        for idx, matrix in enumerate(matrices):
            # rank normalize (and flatten) the matrix
            rank_normalized_matrix_flat = scipy.stats.rankdata(matrix.values)
            if idx > 0:
                # rescale it by the first matrix
                rank_normalized_matrix_flat = np.interp(
                    rank_normalized_matrix_flat,
                    (rank_normalized_matrix_flat.min(), rank_normalized_matrix_flat.max()),
                    (rank_normalized_matrices[0].min(), rank_normalized_matrices[0].max())
                    )
            rank_normalized_matrices.append(
                rank_normalized_matrix_flat.reshape(matrix.shape)
                )
        # fuse the matrices horizontally
        fused_matrix = np.hstack(rank_normalized_matrices)
        # rescale to 0-1 (which also zeros out originally zero values)
        fused_matrix = np.interp(
            fused_matrix.flatten(),
            (fused_matrix.flatten().min(), fused_matrix.flatten().max()), # from
            (0, 1) # to
        ).reshape(fused_matrix.shape)
        fused_matrix = pd.DataFrame(
            fused_matrix,
            index = matrices[0].index,
            columns = matrices[0].index.tolist() * len(matrices)
        )
        return fused_matrix

    def _make_sparse_fairly(self, sparsity):
        """
        Applies sparsity to a fused matrix fairly to the
        matrices so that the same number of nodes are selected
        per region per matrix, and the same number are zeroed out
        """
        n_parcels = self.matrix.shape[0]
        n_matrices = self.matrix.shape[1] // n_parcels
        split_matrices = []
        for i in range(n_matrices):
            split_matrix = self.matrix.copy().iloc[:, n_parcels*(i):n_parcels*(i+1)]
            selected_edges = split_matrix.apply(lambda row: row>=row.quantile(sparsity), axis=1)
            split_matrix[~selected_edges]=0
            split_matrices.append(split_matrix)
        return np.hstack(split_matrices)

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
        sub_dir = f'parc-{self.parcellation_name}'
        if (self.input_type != 'density') & (not self.relative):
            sub_dir += '_absolute'
        if (self.input_type != 'density'):
            sub_dir += f'_curv-{str(self.correct_curvature).lower()}'
        if (self.input_type != 'thickness') & (self.laminar_density):
            sub_dir += '_laminar'
        if self.regress_out_geodesic_distance:
            sub_dir += '_reg_out_gd'
        if self.exc_regions:
            sub_dir += f'_exc-{self.exc_regions}'
        if self.merge_layers:
            sub_dir += f'_merge-{"-".join(["".join(map(str, group)) for group in self.merge_layers])}'
        if self.zero_out_negative:
            sub_dir += f'_zero_negs'
        sub_dir += f'_metric-{self.similarity_metric}'
        return os.path.join(OUTPUT_DIR, main_dir, sub_dir)

    def plot_parcels_profile(self, palette='bigbrain', order=None, axis_off=True, save=False):
        """
        Plots the profile of all parcels in a stacked bar plot

        Parameters
        ---------
        palette: (str)
            - bigbrain: layer colors on BigBrain web viewer
            - wagstyl: layer colors on Wagstyl 2020 paper
        order: (None | list)
            different order of parcels (e.g. sorted by gradient values)
        """
        if not hasattr(self, '_parcellated_input_data'):
            self._load_input_data()
        # remove NaNs and reindex + apply order if indicated
        if order is None:
            concat_parcellated_input_data = self._parcellated_input_data.dropna().reset_index(drop=True)
        else:
            concat_parcellated_input_data = self._parcellated_input_data.loc[order].dropna().reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(100, 20))
        if self.input_type == 'thickness':
            if self.relative:
                concat_parcellated_input_data = concat_parcellated_input_data.divide(
                    concat_parcellated_input_data.sum(axis=1),
                    axis=0
                    )
            # plot the relative thickness of layers 6 to 1
            colors = datasets.LAYERS_COLORS[palette]
            ax.bar(
                x = concat_parcellated_input_data.index,
                height = concat_parcellated_input_data.iloc[:, 5],
                width = 1,
                color=colors[5]
                )
            for col_idx in range(4, -1, -1):
                ax.bar(
                    x = concat_parcellated_input_data.index,
                    height = concat_parcellated_input_data.iloc[:, col_idx],
                    width = 1,
                    bottom = concat_parcellated_input_data.iloc[:, ::-1].cumsum(axis=1).loc[:, col_idx+1],
                    color=colors[col_idx]
                    )
        elif self.input_type == 'density':
            profiles = self._parcellated_input_data.dropna().reset_index(drop=True)
            profiles = profiles / profiles.values.max()
            img_data = ((profiles.values.T) * 255).astype('uint8')
            img = PIL.Image.fromarray(img_data , 'L').resize((1000, 400))
            ax.imshow(img, cmap='bone')
        ax.set_xticklabels(concat_parcellated_input_data.index.tolist(), rotation=90)
        if axis_off:
            ax.axis('off')
        if save:
            fig.tight_layout()
            fig.savefig(os.path.join(self.dir_path,'parcels_profile'), dpi=192)

class CorticalTypeDiffMatrix(Matrix):
    def __init__(self, parcellation_name, exc_regions=None):
        """
        Initializes cortical type difference matrix object for the given `parcellation_name`

        Parameters
        ---------
        parcellation_name: (str)
        exc_regions: (str | None)
        """
        self.parcellation_name = parcellation_name
        self.exc_regions = exc_regions
        self.label = "Cortical type difference"
        self.cmap = 'YlOrRd'
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'ctypes', 
            f'diff_parc-{self.parcellation_name}'\
            + (f'_exc-{self.exc_regions}' if self.exc_regions else '')
            )
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        self._create()
        # convert dtype to category (important for compare_fit function)
        self.matrix = self.matrix.astype('int').astype('category')

    def _create(self):
        """
        Create cortical type difference matrix
        """
        # load cortical types per parcel
        cortical_types = surfaces.CorticalTypes(
            self.exc_regions, 
            parcellation_name=self.parcellation_name)
        parcellated_cortical_types = cortical_types.parcellated_data['Cortical Type']
        # get the magnitude of difference between the cortical types
        ctype_diff = np.abs(
            parcellated_cortical_types.values[:, np.newaxis] \
            - parcellated_cortical_types.values[np.newaxis, :]
            )
        # label the matrix
        self.matrix = pd.DataFrame(
            ctype_diff,
            index=parcellated_cortical_types.index,
            columns=parcellated_cortical_types.index,
        )

class CorrelatedGeneExpressionMatrix(Matrix):
    def __init__(self, parcellation_name='sjh', brain_specific=True):
        """
        Correlated gene expression (CGE) matrix showing how similar is the
        pattern of gene expression between regions

        Parameters
        ---------
        parcellation_name: (str)
        brain_specific: (bool)
            limits the genes to the list of brain-specific genes based on Burt 2018
        create_plots: (bool)
        """
        self.parcellation_name = parcellation_name
        self.label = 'Correlated gene expression'
        self.short_label = 'CGE'
        self.cmap = 'YlOrRd'
        self.dir_path = os.path.join(OUTPUT_DIR, 'cge', f'parc-{parcellation_name}')
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        ahba_df = datasets.fetch_ahba_data(
            parcellation_name=self.parcellation_name, 
            ibf_threshold=0.5, missing='centroids')['all']
        if brain_specific:
            brain_specific_genes_in_ahba = ahba_df.columns.intersection(abagen.fetch_gene_group('brain'))
            ahba_df = ahba_df.loc[:, brain_specific_genes_in_ahba]
        matrix = np.corrcoef(ahba_df.values)
        # zero out correlations of 1 (to avoid division by 0)
        matrix[np.isclose(matrix, 1)] = 0
        # Fisher's z-transformation
        matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
        # zero out NaNs and inf
        matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        self.matrix = pd.DataFrame(matrix, columns=ahba_df.index, index=ahba_df.index)

class StructuralCovarianceMatrix(Matrix):
    def __init__(self):
        """
        Structural covariance matrix obtained from Valk 2020
        (https://doi.org/10.1126/sciadv.abb3417)
        """
        self.parcellation_name = 'schaefer400'
        self.label = 'Structural covariance'
        self.short_label = 'StrCov'
        self.cmap = 'RdBu_r'
        self.dir_path = os.path.join(OUTPUT_DIR, 'strcov')
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        matrix = scipy.io.loadmat(os.path.join(SRC_DIR, 'scov_Valk2020.mat'), simplify_cells=True)['SCOV']['strcov']
        schaefer400_labels = datasets.load_volumetric_parcel_labels('schaefer400')
        self.matrix = pd.DataFrame(matrix, index=schaefer400_labels, columns=schaefer400_labels)

class GeneticCorrelationMatrix(Matrix):
    def __init__(self):
        """
        Genetic correlation matrix obtained from Valk 2020
        (https://doi.org/10.1126/sciadv.abb3417)
        """
        self.parcellation_name = 'schaefer400'
        self.label = 'Genetic correlation'
        self.short_label = 'GenCorr'
        self.cmap = 'RdBu_r'
        self.dir_path = os.path.join(OUTPUT_DIR, 'gencorr')
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        matrix = scipy.io.loadmat(os.path.join(SRC_DIR, 'scov_Valk2020.mat'), simplify_cells=True)['SCOV']['gencorr']
        schaefer400_labels = datasets.load_volumetric_parcel_labels('schaefer400')
        self.matrix = pd.DataFrame(matrix, index=schaefer400_labels, columns=schaefer400_labels)

class EnvironmentalCorrelationMatrix(Matrix):
    def __init__(self):
        """
        Environmental correlation matrix obtained from Valk 2020
        (https://doi.org/10.1126/sciadv.abb3417)
        """
        self.parcellation_name = 'schaefer400'
        self.label = 'Environmental correlation'
        self.short_label = 'EnvCorr'
        self.cmap = 'RdBu_r'
        self.dir_path = os.path.join(OUTPUT_DIR, 'envcorr')
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        matrix = scipy.io.loadmat(os.path.join(SRC_DIR, 'scov_Valk2020.mat'), simplify_cells=True)['SCOV']['envcorr']
        schaefer400_labels = datasets.load_volumetric_parcel_labels('schaefer400')
        self.matrix = pd.DataFrame(matrix, index=schaefer400_labels, columns=schaefer400_labels)


class MaturationalCouplingMatrix(Matrix):
    def __init__(self):
        """
        Group-averaged matruational coupling matrix obtained from 
        Khundrakpam 2019 (https://doi.org/10.1093/cercor/bhx317)
        """
        self.parcellation_name = 'aal'
        self.label = 'Maturational coupling'
        self.short_label = 'MCM'
        self.cmap = 'YlGnBu_r'
        self.dir_path = os.path.join(OUTPUT_DIR, 'mcm')
        self.file_path = os.path.join(self.dir_path, 'matrix')
        os.makedirs(self.dir_path, exist_ok=True)
        mcm_labels = (
            pd.read_csv(os.path.join(SRC_DIR, 'MCM_averaged_aal_labels.tsv'), sep='\t', index_col=0)
            .iloc[:,0]
            .str.replace("'", "")
        )
        matrix = scipy.io.loadmat(os.path.join(SRC_DIR, 'MCM_averaged_aal.mat'), simplify_cells=True)['MCM_average']
        self.matrix = pd.DataFrame(matrix, index=mcm_labels, columns=mcm_labels)