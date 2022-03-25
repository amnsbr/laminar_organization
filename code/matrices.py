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
import statsmodels.stats.multitest
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

class Matrix:
    """
    Generic class for matrices
    """
    split_hem = False # set to False by default and only change for GD and SC
    cmap = 'viridis' # default cmap

    def _save(self):
        """
        Save the matrix to a .csv file
        """
        self.matrix.to_csv(
            self.file_path, 
            index_label='parcel'
        )

    def _load(self):
        """
        Loads a matrix created before
        """
        self.matrix = pd.read_csv(
            self.file_path,
            index_col='parcel'
        )
        # for sjh parcellation index is read as 'int' but columns
        # are read as str. Fix this by making columns similar to
        # index * number of matrices
        n_sq_matrix = self.matrix.shape[1] // self.matrix.shape[0]
        self.matrix.columns = self.matrix.index.tolist() * n_sq_matrix

    def _remove_parcels(self, exc_mask_type):
        """
        Removes the allocortex +/- adysgranular parcels from the matrix (after the matrix is created)
        
        Note: This should be used for the matrices other than MicrostructuralCovarianceMatrix
        as in that case the adysgranular regions are removed from the input data (this 
        is important when using partial correlation)

        exc_mask_type: (str)
            - allocortex: excludes allocortex
            - adysgranular: excludes allocortex + adysgranular regions
        """
        exc_masks = datasets.load_exc_masks(exc_mask_type, self.parcellation_name)
        parcels_to_exclude = helpers.parcellate(exc_masks, self.parcellation_name)
        parcels_to_exclude = helpers.concat_hemispheres(parcels_to_exclude, dropna=True)
        parcels_to_include = (1 - parcels_to_exclude).astype('bool')
        parcels_to_include_idx = parcels_to_include[parcels_to_include.values].index
        return self.matrix.loc[parcels_to_include_idx, parcels_to_include_idx]

    def plot(self, vrange=(0.025, 0.975)):
        """
        Plot the matrix as heatmap
        """
        #> plot the matrix
        helpers.plot_matrix(
            self.matrix.values,
            os.path.join(self.file_path.replace('.csv','')),
            cmap=self.cmap,
            vrange=vrange
            )

    def correlate_edge_wise(self, other, test='pearson'):
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
        """
        # TODO: maybe move this to MicrostructuralCovarianceMatrix
        #> make sure they have the same parcellation and mask
        assert self.parcellation_name == other.parcellation_name
        assert self.exc_adys == other.exc_adys
        #> match the matrices in the order and selection of parcels
        #  + convert them to np.ndarray
        shared_parcels = self.matrix.index & other.matrix.index # union of the two indices
        X = self.matrix.loc[shared_parcels, shared_parcels].values
        Y = other.matrix.loc[shared_parcels, shared_parcels].values
        # #> make NaNs zero so NaN can be assigned to L-R edges
        # X[np.isnan(X)] = 0 # TODO: make sure this is okay for all the matrices
        # Y[np.isnan(Y)] = 0
        #> if L and R should be investigated separately
        # make interhemispheric pairs of the lower triangle
        # NaN so it could be then removed
        if self.split_hem or other.split_hem:
            exc_regions = 'adysgranular' if self.exc_adys else 'allocortex'
            split_hem_idx = helpers.get_split_hem_idx(
                self.parcellation_name, 
                exc_regions
                )
            X[split_hem_idx:, :split_hem_idx] = np.NaN
            Y[split_hem_idx:, :split_hem_idx] = np.NaN
        #> get the index for lower triangle
        tril_index = np.tril_indices_from(X, -1)
        x = X[tril_index]
        y = Y[tril_index]
        #> remove NaNs (e.g. interhemispheric pairs) and 0s
        #  as the Y matrix is often zeroed out and 0
        #  has lost its meaning and there's a lot of zeros which
        #  have been actually the negative values
        mask = ~(x==0 | np.isnan(x) | (y==0) | np.isnan(y)) 
        x = x[mask]
        y = y[mask]
        #> correlation
        if test == 'pearson':
            coef, p_val = scipy.stats.pearsonr(x, y)
        else:
            coef, p_val = scipy.stats.spearmanr(x, y)
        res_str = f"{test.title()} correlation with {other.label}\nCoef: {coef}; Parametric p-value: {p_val}"
        # TODO: calculate p-value non-parametrically?
        # TODO: regress out GD in some cases
        out_path = os.path.join(self.dir_path, f'correlation_{other.label.lower().replace(" ", "_")}')
        with open(out_path+'.txt', 'w') as res_file:
            res_file.write(res_str)
        #> plotting
        jp = sns.jointplot(
            x = x, y = y, kind = "hex", 
            color = "grey", height = 4,
            marginal_kws = {'bins':35}, 
            joint_kws = {'gridsize':35},
            # xlim = (np.quantile(X[tril_index], 0.025), np.quantile(X[tril_index], 0.975)),
            # ylim = (np.quantile(Y[tril_index], 0.025), np.quantile(Y[tril_index], 0.975)),
        )
        ax = jp.ax_joint
        sns.regplot(
            X[tril_index], Y[tril_index], 
            ax=ax, ci=None, scatter=False, 
            color='C0', line_kws=dict(alpha=0.2)
            )
        #> add rho on the figure
        text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
        text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.95
        if test == 'pearson':
            text = f'r = {coef:.2f}'
        else:
            text = f'rho = {coef:.2f}'
        ax.text(text_x, text_y, text,
                color='black', size=16,
                multialignment='left')
        ax.set_xlabel(self.label)
        ax.set_ylabel(other.label)
        jp.fig.tight_layout()
        jp.fig.savefig(out_path+'.png', dpi=192)
        # #> plotting half-half matrix
        # uhalf_X = X.copy()
        # uhalf_X[np.tril_indices_from(X, 0)] = np.NaN
        # lhalf_Y = Y.copy()
        # lhalf_Y[np.triu_indices_from(Y, 0)] = np.NaN
        # fig, ax = plt.subplots(figsize=(7,7))
        # sns.heatmap(
        #     uhalf_X,
        #     vmin=np.quantile(uhalf_X.flatten(),0.05),
        #     vmax=np.quantile(uhalf_X.flatten(),0.95),
        #     cbar=False,
        #     ax=ax)
        # ax.axis('off')

    def correlate_node_wise(self, other, test='pearson', plot=False):
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
        """
        #> make sure they have the same parcellation and mask
        assert self.parcellation_name == other.parcellation_name
        assert self.exc_adys == other.exc_adys
        #> match the matrices in the order and selection of parcels
        #  + convert them to np.ndarray
        shared_parcels = self.matrix.index & other.matrix.index # union of the two indices
        X = self.matrix.loc[shared_parcels, shared_parcels].values
        Y = other.matrix.loc[shared_parcels, shared_parcels].values
        #> if L and R should be investigated separately
        # make interhemispheric pairs of the lower triangle
        # NaN so it could be then removed
        if self.split_hem or other.split_hem:
            exc_regions = 'adysgranular' if self.exc_adys else 'allocortex'
            split_hem_idx = helpers.get_split_hem_idx(
                self.parcellation_name, 
                exc_regions
                )
            X[split_hem_idx:, :split_hem_idx] = np.NaN
            Y[split_hem_idx:, :split_hem_idx] = np.NaN
        #> calculate the correlation at each row (node)
        node_coefs = pd.Series(np.empty(X.shape[0]), index=shared_parcels)
        node_pvals = pd.Series(np.empty(X.shape[0]), index=shared_parcels)
        for row_idx in range(X.shape[0]):
            row_x = X[row_idx, :]
            row_y = Y[row_idx, :]
            #> remove NaNs and 0s
            row_mask = ~((row_x==0) | np.isnan(row_x) | (row_y==0) | np.isnan(row_y)) 
            row_x = row_x[row_mask]
            row_y = row_y[row_mask]
            #> calculate correlation
            if test == 'pearson':
                node_coefs.iloc[row_idx], node_pvals.iloc[row_idx] = scipy.stats.pearsonr(row_x, row_y)
            else:
                node_coefs.iloc[row_idx], node_pvals.iloc[row_idx] = scipy.stats.spearmanr(row_x, row_y)
        #> FDR correction
        _, node_pvals_fdr = statsmodels.stats.multitest.fdrcorrection(node_pvals)
        node_pvals_fdr = pd.Series(node_pvals_fdr, index=shared_parcels)
        node_coefs_sig = node_coefs.copy()
        node_coefs_sig.loc[node_pvals_fdr >= 0.05] = 0
        #> project to surface
        node_coefs_surface = helpers.deparcellate(node_coefs, self.parcellation_name)
        node_coefs_sig_surface = helpers.deparcellate(node_coefs_sig, self.parcellation_name)
        #> save (and plot) surface maps
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
            helpers.plot_on_bigbrain_nl(
                node_coefs_sig_surface,
                filename= out_path + '_nodewise_surface_sig.png'
            )

    def associate_cortical_types(self):
        """
        Calculates within and between cortical type average of matrix 
        values and plots a collapsed matrix by cortical types. 
        Excludes a-/dysgranular as indicated by the matrix

        Returns
        ---------
        matrix_file: (str) path to matrix file    
        """
        #> load cortical types and make it match matrix index
        parcellated_cortical_types = datasets.load_cortical_types(self.parcellation_name)
        parcellated_cortical_types = parcellated_cortical_types.loc[self.matrix.index]
        #> list the excluded types based on matrix path 
        if self.exc_adys:
            included_types = ['EU1', 'EU2', 'EU3', 'KO']
            excluded_types = [np.NaN, 'ALO', 'AG', 'DG']
        else:
            included_types = ['AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO']
            excluded_types = [np.NaN, 'ALO']
        n_types = len(included_types)
        matrix = self.matrix.loc[
            ~parcellated_cortical_types.isin(excluded_types), 
            ~parcellated_cortical_types.isin(excluded_types)
            ] # this is probably not needed since excluded types are already excluded!
        parcellated_cortical_types = (
            parcellated_cortical_types
            .loc[~parcellated_cortical_types.isin(excluded_types)]
            .cat.remove_unused_categories()
            )
        #> collapse matrix to mean values per cortical type
        mean_matrix_by_cortical_type = pd.DataFrame(np.zeros((n_types, n_types)),
                                                    columns=included_types,
                                                    index=included_types)
        for group_idx, group_df in matrix.groupby(parcellated_cortical_types):
            mean_matrix_by_cortical_type.loc[group_idx, :] = \
                (group_df
                .T.groupby(parcellated_cortical_types) # group columns by cortical type
                .mean() # take average of cortical types for each row (parcel)
                .mean(axis=1)) # take the average of average of rows in each cortical type
        #> plot it
        helpers.plot_matrix(
            mean_matrix_by_cortical_type.values,
            self.file_path.replace('.csv', '_averaged-ctypes'),
            vrange=(0, 1), cmap=self.cmap
            )
        #> quantify intra and intertype similarity
        intra_intertype = pd.DataFrame(
            np.zeros((n_types, 2)),
            columns=['intra', 'inter'],
            index=included_types
            )
        for group_idx, group_df in matrix.groupby(parcellated_cortical_types):
            intra_intertype.loc[group_idx, 'intra'] = \
                (group_df
                .T.groupby(parcellated_cortical_types) # group columns by cortical type
                .mean() # take average of cortical types for each row (parcel)
                .loc[group_idx].mean()) # take the average of average of rows in each cortical type
            intra_intertype.loc[group_idx, 'inter'] = \
                (group_df
                .T.groupby(parcellated_cortical_types) # group columns by cortical type
                .mean() # take average of cortical types for each row (parcel)
                .drop(index=group_idx) # remove the same type
                .values.mean()) # take the average of average of rows in each cortical type
        #> test significance using permutation
        null_dist_intra_intertype = np.zeros((1000, n_types, 2))
        for perm_idx in range(1000):
            #> create a surrogate shuffled matrix
            shuffled_parcels = np.random.permutation(matrix.index.tolist())
            surrogate = matrix.loc[shuffled_parcels, shuffled_parcels]
            surrogate.index = matrix.index
            surrogate.columns = matrix.columns
            null_intra_intertype = pd.DataFrame(
                np.zeros((n_types, 2)),
                columns=['intra', 'inter'],
                index=included_types)
            for group_idx, group_df in surrogate.groupby(parcellated_cortical_types):
                null_intra_intertype.loc[group_idx, 'intra'] = \
                    (group_df
                    .T.groupby(parcellated_cortical_types) # group columns by cortical type
                    .mean() # take average of cortical types for each row (parcel)
                    .loc[group_idx].mean()) # take the average of average of rows in each cortical type
                null_intra_intertype.loc[group_idx, 'inter'] = \
                    (group_df
                    .T.groupby(parcellated_cortical_types) # group columns by cortical type
                    .mean() # take average of cortical types for each row (parcel)
                    .drop(index=group_idx) # remove the same type
                    .dropna() # drop unknown and ALO
                    .values.mean()) # take the average of average of rows in each cortical type
            null_dist_intra_intertype[perm_idx, :, :] = null_intra_intertype.values
        null_dist_diff_intra_inter = null_dist_intra_intertype[:, :, 0] - null_dist_intra_intertype[:, :, 1]
        diff_intra_inter = (intra_intertype.iloc[:, 0] - intra_intertype.iloc[:, 1]).values.reshape(1, -1)
        intra_intertype['pvals'] = (null_dist_diff_intra_inter > diff_intra_inter).mean(axis=0)
        intra_intertype.to_csv(self.file_path.replace('.csv', '_intra_intertype_diff.txt'))

class CurvatureSimilarityMatrix(Matrix):
    """
    Matrix showing similarity of curvature distribution
    between each pair of parcels
    """
    label = "Curvature similarity"
    dir_path = os.path.join(OUTPUT_DIR, 'curvature')
    cmap = sns.color_palette("mako", as_cmap=True)
    def __init__(self, parcellation_name, exc_adys=True):
        """
        Creates curvature similarity matrix or loads it if it already exist

        Parameters
        ----------
        parcellation_name: (str)
        exc_adys: (bool)
        """
        self.parcellation_name = parcellation_name
        self.exc_adys = exc_adys
        file_name =  f'curvature_similarity_matrix_parc-{parcellation_name}.csv'
        self.file_path = os.path.join(self.dir_path, file_name)
        if os.path.exists(self.file_path):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self.matrix = self._create()
            self._save()
        #> remove the adysgranular parcels after the matrix is created
        #  to avoid unncessary waste of computational resources
        #  note this is not saved
        if self.exc_adys:
            self.matrix = self._remove_parcels('adysgranular')
        else:
            self.matrix = self._remove_parcels('allocortex')
        self.plot()
        
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
        return curv_similarity_matrix

class DistanceMatrix(Matrix):
    """
    Matrix of geodesic/Euclidean distance between centroids of parcels
    """
    dir_path = os.path.join(OUTPUT_DIR, 'distance')
    split_hem = True
    cmap = sns.color_palette("viridis", as_cmap=True)
    def __init__(self, parcellation_name, kind='geodesic', 
                 approach='center-to-center', 
                 exc_adys=True, keep_allo=False):
        """
        Initializes geodesic distance matrix for `parcellation_name` and
        creates or loads it

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
        exc_adys: (bool) Exclude adysgranular parcels from the matrix
        keep_allo: (bool) Keep allocortical regions. Only used if exc_adys=False.
        Based on "geoDistMapper.py" from micapipe/functions
        Original Credit:
        # Translated from matlab:
        # Original script by Boris Bernhardt and modified by Casey Paquola
        # Translated to python by Jessica Royer
        """
        self.parcellation_name = parcellation_name
        self.kind = kind
        self.approach = approach
        print(self.approach)
        self.exc_adys = exc_adys
        self.label = f"{self.kind.title()} distance"
        self.file_path = os.path.join(
            self.dir_path,
            f'{self.kind}_distance_matrix_parc-{self.parcellation_name}_approach-{self.approach}.csv'
            )
        if os.path.exists(self.file_path):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            if self.kind == 'geodesic':
                self.matrix = self._create_geodesic()
            else:
                self.matrix = self._create_euclidean()
            self._save()
            self.plot()
        #> remove the adysgranular parcels after the matrix is created
        #  to avoid unncessary waste of computational resources
        #  note this is not saved or plotted
        if self.exc_adys:
            self.matrix = self._remove_parcels('adysgranular')
        elif not keep_allo:
            self.matrix = self._remove_parcels('allocortex') 

    def _create_geodesic(self):
        """
        Creates center-to-parcel or center-to-center geodesic distance matrix
        between pairs  of parcels
        """
        GDs = {}
        for hem in ['L', 'R']:
            #> load surf
            surf_path = os.path.join(
                SRC_DIR, 
                f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                )
            surf = nibabel.load(surf_path)
            vertices = surf.agg_data('NIFTI_INTENT_POINTSET')            
            parc = datasets.load_parcellation_map(self.parcellation_name, False)[hem]
            # find centre vertices
            uparcel = np.unique(parc)
            voi = np.zeros([1, len(uparcel)])
            
            print(f"[ INFO ]..... Finings centre vertex for each parcel in hemisphere {hem}")
            # TODO: use helpers.get_parcel_centers instead
            for (i, parcel_name) in enumerate(uparcel):
                this_parc = np.where(parc == parcel_name)[0]
                if this_parc.size == 1: # e.g. L_unknown in aparc
                    voi[0, i] = this_parc[0]
                else:
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
                cmdStr = f"{wbPath} -surface-geodesic-distance {surf_path} {center_vertex} {self.file_path}_this_voi.func.gii"
                subprocess.run(cmdStr.split())
                tmpname = self.file_path + '_this_voi.func.gii'
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
            # #> save the GD for the current hemisphere
            # np.savetxt(
            #     self.file_path.replace('_parc', f'_hemi-{hem}_parc'),
            #     GDs[hem],
            #     fmt='%.12f')
            #> convert it to dataframe so that joining hemispheres would be easier
            GDs[hem] = pd.DataFrame(GDs[hem], index=uparcel, columns=uparcel)
        #> join the GD matrices from left and right hemispheres
        GD_full = (pd.concat([GDs['L'], GDs['R']],axis=0)
                .reset_index(drop=False)
                .drop_duplicates('index')
                .set_index('index')
                .fillna(0))
        return GD_full

    def _create_euclidean(self):
        center_indices = helpers.get_parcel_center_indices(self.parcellation_name)
        center_coords = {}
        for hem in ['L', 'R']:
            surf_path = os.path.join(
                SRC_DIR, 
                f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
                )
            coords = nilearn.surface.load_surf_mesh(surf_path).coordinates
            center_coords[hem] = coords[center_indices[hem].values]
        center_coords = np.vstack([center_coords['L'], center_coords['R']])
        ED_matrix = scipy.spatial.distance_matrix(center_coords, center_coords)
        parcels = center_indices['L'].index.to_list() + center_indices['R'].index.to_list()
        ED_matrix = pd.DataFrame(ED_matrix, columns=parcels, index=parcels)
        return ED_matrix

class ConnectivityMatrix(Matrix):
    """
    Structural or functional connectivity matrix
    """
    def __init__(self, kind, exc_adys=True, sc_zero_contra=True, parcellation_name='schaefer400'):
        """
        Initializes structural/functional connectivity matrix

        Parameters
        ---------
        kind: (str)
            - structural
            - functional
        exc_adys: (bool) exclude adysgranular parcels
        sc_zero_contra: (bool) zero out contralateral edges for the SC matrix
        parcellation_name: (str)
            - schaefer400 (currently only this is supported)
        """
        self.kind = kind
        self.exc_adys = exc_adys
        self.sc_zero_contra = sc_zero_contra
        if self.kind == 'structural':
            self.cmap = cmcrameri.cm.davos
            if self.sc_zero_contra:
                self.split_hem = True
        else:
            self.cmap = cmcrameri.cm.acton
        self.parcellation_name = parcellation_name
        self.label = f'{self.kind.title()} connectivity'
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'connectivity',
            f'{self.kind[0]}c'\
            + f'_parc-{self.parcellation_name}' \
            + ('_split_hem' if self.split_hem else '')
            )
        self.file_path = os.path.join(self.dir_path, 'matrix.csv')
        if os.path.exists(self.file_path):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self.matrix = datasets.load_conn_matrix(self.kind, self.parcellation_name)
            if (self.kind == 'structural') & self.sc_zero_contra:
                exc_regions = 'adysgranular' if self.exc_adys else 'allocortex'
                split_hem_idx = helpers.get_split_hem_idx(
                    self.parcellation_name, 
                    exc_regions
                    )
                self.matrix.iloc[:split_hem_idx, split_hem_idx:] = 0
                self.matrix.iloc[split_hem_idx:, :split_hem_idx] = 0
            self._save()
        #> remove the adysgranular parcels after the matrix is created
        #  to avoid unncessary waste of computational resources
        #  note this is not saved
        if self.exc_adys:
            self.matrix = self._remove_parcels('adysgranular')
        else:
            self.matrix = self._remove_parcels('allocortex') 
        self.plot()


class MicrostructuralCovarianceMatrix(Matrix):
    """
    Matrix showing microstructural similarity of parcels in their relative laminar
    thickness, relative laminar volume, density profiles (MPC), or their combination
    """
    def __init__(self, input_type, parcellation_name='sjh', 
                 exc_adys=True, correct_curvature='volume', 
                 similarity_metric = 'parcor', similarity_scale='parcel'):
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
        exc_adys: (bool) Exclude a-/dysgranular regions
        parcellation_name: (str) Parcellation scheme
            - 'sjh'
            - 'schaefer400'
        """
        #> save parameters as class fields
        self.input_type = input_type
        self.correct_curvature = correct_curvature
        self.similarity_metric = similarity_metric
        self.similarity_scale = similarity_scale
        self.parcellation_name = parcellation_name
        self.exc_adys = exc_adys
        #> set the label based on input type
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
        self.short_label = SHORT_LABELS[self.input_type]
        #> set cmap
        if self.input_type == 'density':
            self.cmap = sns.color_palette("rocket", as_cmap=True)
        else: # TODO: use different colors for each input type in the plot in thickness-density
            self.cmap = sns.color_palette("RdBu_r", as_cmap=True)
        #> directory and filename (prefix which will be used for .npz and .jpg files)
        self.dir_path = self._get_dir_path()
        os.makedirs(self.dir_path, exist_ok=True)
        self.file_path = os.path.join(self.dir_path, 'matrix.csv')
        if os.path.exists(self.file_path):
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
            - exc_adys: {self.exc_adys}
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
                    exc_adys=self.exc_adys,
                    parcellation_name=self.parcellation_name
                )
                matrices.append(matrix_obj.matrix)
                #TODO: it is unlikely but maybe valid parcels are different for each modality
                # in that case this code won't work properly
            self.matrix = self._fuse_matrices(matrices)
        else:
            #> Load laminar thickness or density profiles
            print("Reading input")
            self._input_data = self._load_input_data()
            #> create the similarity matrix
            if self.similarity_scale == 'parcel':
                self.matrix = self._create_at_parcels()
            else:
                self.matrix = self._create_at_vertices()

    def _load_input_data(self):
        """
        Load input data
        """
        #> load exclusion masks for a-/dysgranular regions if needed
        if self.exc_adys:
            exc_masks = datasets.load_exc_masks('adysgranular', self.parcellation_name)
        else:
            exc_masks = datasets.load_exc_masks('allocortex', self.parcellation_name)
        #> load the data
        if self.input_type == 'thickness':
            if self.correct_curvature == 'volume':
                input_data = datasets.load_laminar_volume(
                    exc_masks=exc_masks,
                )
            elif self.correct_curvature == 'regress':
                input_data = datasets.load_laminar_thickness(
                    exc_masks=exc_masks,
                    regress_out_curvature=True,
                    normalize_by_total_thickness=True,
                )
            else:
                input_data = datasets.load_laminar_thickness(
                    exc_masks=exc_masks,
                    regress_out_curvature=False,
                    normalize_by_total_thickness=True,
                )
        elif self.input_type == 'density':
            input_data = datasets.load_total_depth_density(
                exc_masks=exc_masks
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
            matrix[np.isclose(matrix, 1)] = 0
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
        matrix = pd.DataFrame(
            matrix, 
            index=valid_parcels,
            columns=valid_parcels)
        return matrix

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
        matrix = pd.DataFrame(
            matrix, 
            index=valid_parcels,
            columns=valid_parcels)
        return matrix

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
        for idx, matrix in enumerate(matrices):
            #> rank normalize (and flatten) the matrix
            rank_normalized_matrix_flat = scipy.stats.rankdata(matrix.values)
            if idx > 0:
                #> rescale it by the first matrix
                rank_normalized_matrix_flat = np.interp(
                    rank_normalized_matrix_flat,
                    (rank_normalized_matrix_flat.min(), rank_normalized_matrix_flat.max()),
                    (rank_normalized_matrices[0].min(), rank_normalized_matrices[0].max())
                    )
            rank_normalized_matrices.append(
                rank_normalized_matrix_flat.reshape(matrix.shape)
                )
        #> fuse the matrices horizontally
        fused_matrix = np.hstack(rank_normalized_matrices)
        #> rescale to 0-1 (which also zeros out originally zero values)
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
        #TODO this is called from MicrostructuralCovarianceGradients. Fix it!
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
        if self.input_type != 'density':
            sub_dir += f'_curv-{str(self.correct_curvature).lower()}'
        if self.exc_adys:
            sub_dir += '_exc-adys'
        sub_dir += f'_metric-{self.similarity_metric}'\
            + f'_scale-{self.similarity_scale}'
        return os.path.join(OUTPUT_DIR, main_dir, sub_dir)

class CorticalTypeSimilarityMatrix(Matrix):
    label = "Cortical type similarity"
    cmap = 'YlOrRd'
    def __init__(self, parcellation_name, exc_adys):
        """
        Initializes cortical type similarity matrix object for the given `parcellation_name`

        Parameters
        ---------
        parcellation_name: (str)
        exc_adys: (bool) exclude adysgranular+allocortical regions or only allocortex
        """
        self.parcellation_name = parcellation_name
        self.exc_adys = exc_adys
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'ctypes', 
            f'similarity_parc-{self.parcellation_name}'\
            + ('_exc-adys' if self.exc_adys else '')
            )
        self.file_path = os.path.join(self.dir_path, 'matrix.csv')
        if os.path.exists(self.file_path):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self._create()
            if self.exc_adys:
                self.matrix = self._remove_parcels('adysgranular')
            else:
                self.matrix = self._remove_parcels('allocortex') 
            self._save()
            self.plot()            

    def _create(self):
        """
        Create cortical type similarity matrix
        """
        #> load cortical types per parcel
        parcellated_cortical_types = datasets.load_cortical_types(self.parcellation_name)
        parcellated_cortical_types = parcellated_cortical_types.dropna()
        #> get the magnitude of difference between the cortical types
        ctype_diff = np.abs(
            parcellated_cortical_types.cat.codes.values[:, np.newaxis] \
            - parcellated_cortical_types.cat.codes.values[np.newaxis, :]
            )
        #> reverse it to make it a similarity matrix in the range of (1, 7)
        ctype_similarity = 7 - ctype_diff
        self.matrix = pd.DataFrame(
            ctype_similarity,
            index=parcellated_cortical_types.index,
            columns=parcellated_cortical_types.index,
        )

class DiseaseCovarianceMatrix(Matrix):
    """
    Cortical thickness co-alteration across disorders
    using ENIGMA toolbox

    Credit: Based on Hettwer 2022 medRxiv
    """
    label = "Disease co-alteration"
    short_label = "DisCov"
    cmap = 'Reds'
    def __init__(self, parcellation_name='aparc', exc_regions='adys', 
                 psych_only=False):
        """
        Initializes the disease covariance matrix
        
        Parameters
        ----------
        parcellation_name: (str)
            the thickness data is only available in 'aparc' but
            can be pseudo-reparcellated to other parcellations
        exc_regions: (str)
            - 'adys': exclude allocortical and adysgranular regions
            - 'allo': exclude allocortical regions
            - None: exclude only the midline
        psych_only: (bool)
            only include psychiatric disorders similar to Hettwer 2022
        """
        self.parcellation_name = parcellation_name
        self.exc_regions = exc_regions
        self.psych_only = psych_only
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'disease', 
            f'covariance_parc-{self.parcellation_name}'\
            + (f'_exc-{self.exc_regions}' if self.exc_regions else '')\
            + ('_psych_only' if self.psych_only else '')
            )
        self.file_path = os.path.join(self.dir_path, 'matrix.csv')
        if os.path.exists(self.file_path):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self.matrix = self._create()
            if self.exc_regions == 'adys':
                self.matrix = self._remove_parcels('adysgranular')
            elif self.exc_regions == 'allo':
                self.matrix = self._remove_parcels('allocortex') 
            self._save()
            self.plot(vrange=(0, 1))
    
    def _create(self):
        """
        Creates disease cortical thickness co-alteration matrix by
        taking the pairwise correlation coefficients across 6-/7-element
        vectors of cortical thickness Cohen's d in each disorder between
        pairs of parcels
        """
        #> load the original data from ENIGMA toolbox
        aparc_disease_maps = datasets.load_disease_maps(self.psych_only)
        #> reparcellated data to target parcellation
        # Note: doing this also with 'aparc' as the target parcellation
        # so the order of parcels is the same across all different types of
        # matrices
        self._input_data = helpers.deparcellate(aparc_disease_maps, 'aparc')
        self._parcellated_input_data = helpers.parcellate(
            self._input_data, self.parcellation_name,
            averaging_method='mean')
        #> calculate the correlation
        matrix = np.corrcoef(self._parcellated_input_data.values)
        #> zero out correlations of 1 (to avoid division by 0)
        # Note: not zeroing out negative values in line with
        # the original paper
        # Note 2: in the original paper no Z-transformation
        # was performed
        matrix[np.isclose(matrix, 1)] = 0
        #> Fisher's z-transformation
        matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
        #> zero out NaNs and inf
        matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        #> convert to df
        matrix = pd.DataFrame(
            matrix, 
            index=self._parcellated_input_data.index,
            columns=self._parcellated_input_data.index
            )
        #> remove midline parcels in the case of exc_regions=None
        # they will cause problems for gradients because
        # all cells in their rows is zero
        midline = matrix.index.isin(helpers.MIDLINE_PARCELS[self.parcellation_name])
        matrix = matrix.loc[~midline, ~midline]
        return matrix

class NeuronalSubtypesCovarianceMatrix(Matrix):
    """
    Matrix showing similarity of gene expression pattern 
    of excitatory and inhibitory neuronal subtypes across
    brain regions using AHBA data and based on gene lists
    from Lake 2018 (10.1038/nbt.4038)
    """
    def __init__(self, neuron_type, parcellation_name, discard_rh=True):
        """
        Creates/loads the matrix

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
            'exc': 'Excitatory neuron subtypes gene expression covariance',
            'inh': 'Inhibitory neuron subtypes gene expression covariance',
        }
        SHORT_LABELS = {
            'exc': 'ExSubCov',
            'inh': 'InSubCov'
        }
        self.label = LABELS.get(self.neuron_type)
        self.short_label = SHORT_LABELS.get(self.neuron_type)
        self.dir_path = os.path.join(
            OUTPUT_DIR, 'ei', 'gene_expression',
            f'{neuron_type}_subtypes_covariance'\
            + f'_parc-{self.parcellation_name}'\
            + ('_rh_discarded' if discard_rh else '')
            )
        self.file_path = os.path.join(self.dir_path, 'matrix.csv')
        if os.path.exists(self.file_path):
            self._load()
        else:
            os.makedirs(self.dir_path, exist_ok=True)
            self.matrix = self._create()
            self._save()
            self.plot(vrange=(0, 1))

    def _create(self):
        """
        Creates the matrix of neuron-subtype-specific
        gene expression covariance across regions by
        taking pair-wise pearson correlation of 
        mean gene expresion patterns

        Returns
        -------
        matrix: (pd.DataFrame) n_parc x n_parc 
        """
        #> load mean gene expression of each parcel-subtype
        self._parcellated_input_data = self._load_input_data()
        #> create matrix of correlations
        matrix = np.corrcoef(self._parcellated_input_data.values)
        #> zero out negative correlations
        matrix[matrix<0] = 0
        #> zero out correlations of 1 (to avoid division by 0)
        matrix[np.isclose(matrix, 1)] = 0
        #> Fisher's z-transformation
        matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
        #> zero out NaNs and inf
        matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        matrix = pd.DataFrame(
            matrix, 
            index=self._parcellated_input_data.index,
            columns=self._parcellated_input_data.index
            )
        return matrix

    def _load_input_data(self):
        """
        Loads the gene list associated with each subtype
        and returns the average expression of genes associated
        with each subtype at each parcel

        Returns
        -------
        subtypes_mean_expression: (pd.DataFrame) n_parc x n_subtypes
        """
        if self.neuron_type == 'exc':
            subtypes = [
                'Ex1', 'Ex2', 'Ex3a', 'Ex3b', 'Ex3c', 
                'Ex3d', 'Ex3e', 'Ex4', 'Ex5a', 'Ex5b', 
                'Ex6a', 'Ex6b', 'Ex8'
                ]
        else:
            subtypes = ['In1a', 'In1b', 'In1c', 'In2', 'In3', 'In4a', 'In4b', 'In6a', 'In6b', 'In7', 'In8']
        subtypes_genes = pd.read_csv(
            os.path.join(
                SRC_DIR, 'neuronal_subtypes_genes.csv'
            ), index_col=0
            )['Cluster']
        subtypes_mean_expression = pd.DataFrame()
        for subtype in subtypes:
            print(subtype)
            #> get the genes list for the subtype
            genes_list = subtypes_genes[subtypes_genes == subtype].index.tolist()
            #> get their mean expression from ahba
            subtypes_mean_expression.loc[:, subtype] = datasets.fetch_mean_gene_expression(
                genes_list, 
                self.parcellation_name, 
                self.discard_rh
                )
            subtypes_mean_expression = subtypes_mean_expression.dropna()
        return subtypes_mean_expression