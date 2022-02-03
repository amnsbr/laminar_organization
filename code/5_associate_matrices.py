"""
Calculate the association between laminar similarity matrices and:
1. FC matrix
2. SC matrix
3. GD matrix (only ipsilateral)
4. curvature similarity matrix
5. MPC matrix
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import enigmatoolbox.datasets
import cmcrameri.cm
import statsmodels.stats.multitest

import helpers
import datasets


#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')

#> config
DO_NODE_WISE = [
    'Functional connectivity', 
    'Structural connectivity', 
    'Laminar thickness similarity'
]
CMAPS = {
    'Geodesic distance': 'viridis',
    'Curvature similarity': sns.color_palette("mako", as_cmap=True),
    'Microstructural profile covariance': 'rocket',
    'Functional connectivity': cmcrameri.cm.acton,
    'Structural connectivity': cmcrameri.cm.davos,
    'Laminar thickness similarity': 'RdBu_r'
}


def load_mpc_matrix(parcellation_name):
    out_path = os.path.join(
        DATA_DIR, 'matrix',
        f'MPC_hist_parc-{parcellation_name}.csv'
    )
    if os.path.exists(out_path):
        return pd.read_csv(out_path, index_col='parcel')
    #> load and parcellate profiles
    profiles = np.loadtxt(os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_desc-profiles.txt'))
    parcellated_profiles = helpers.parcellate(profiles.T, parcellation_name, 'median')
    #> calculate partial correlation
    r_ij = np.corrcoef(parcellated_profiles)
    mean_profile = parcellated_profiles.mean(axis=0)
    r_ic = parcellated_profiles\
                .corrwith(mean_profile, 
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
    #> save the file
    matrix = pd.DataFrame(
        matrix, 
        index=parcellated_profiles.index, 
        columns=parcellated_profiles.index)
    matrix.to_csv(out_path, index_label='parcel')
    return matrix

def load_conn_matrices(matrix_file):
    """
    Loads FC and SC matrices in Schaefer parcellation (400) from ENIGMA toolbox 
    and reorders it according to `matrix_file`. For SC matrix also makes contralateral
    values 0 (so that they are not included in correlations)

    Parameters
    ----------
    matrix_file: (str) path to the matrix .csv file

    Returns
    ---------
    reordered_fc_ctx, reordered_sc_ctx: (np.ndarray) (n_parc, n_parc) 
        reordered FC and SC matrices matching the original matrix
    """
    #> load laminar similarity matrix for its correctly ordered index
    laminar_sim_matrix = pd.read_csv(matrix_file, index_col='parcel')
    #> FC
    fc_ctx, fc_ctx_labels, _, _ = enigmatoolbox.datasets.load_fc('schaefer_400')
    fc_ctx_df = pd.DataFrame(fc_ctx, columns=fc_ctx_labels, index=fc_ctx_labels)
    reordered_fc_ctx = fc_ctx_df.loc[laminar_sim_matrix.index, laminar_sim_matrix.index]
    #> SC
    sc_ctx, sc_ctx_labels, _, _ = enigmatoolbox.datasets.load_sc('schaefer_400')
    sc_ctx_df = pd.DataFrame(sc_ctx, columns=sc_ctx_labels, index=sc_ctx_labels)
    reordered_sc_ctx = sc_ctx_df.loc[laminar_sim_matrix.index, laminar_sim_matrix.index]
    #> zero out SC contralateral
    if ('excmask-adys' in matrix_file):
        parcels_adys = datasets.load_parcels_adys('schaefer400')
        split_hem_idx = int(np.nansum(1-parcels_adys['L'].values)) # number of non-adys parcels in lh
    else:
        split_hem_idx = 200
    reordered_sc_ctx.iloc[:split_hem_idx, split_hem_idx:] = 0
    reordered_sc_ctx.iloc[split_hem_idx:, :split_hem_idx] = 0
    return reordered_fc_ctx, reordered_sc_ctx

def correlate_matrices_edge_wise(X, Y, prefix, xlabel, ylabel, split_hem_idx=None, nonpar=False):
    """
    Calculates and plots the correlation between the edges of two matrices
    `X` and `Y` which are assumed to be square and symmetric. The correlation
    is calculated for lower triangle (excluding the diagonal)

    Paremeters
    ---------
    X, Y: (pd.DataFrame) n_parc x n_parc
    prefix: (str) prefix for the output filename
    xlabel, ylabel: (str)
    split_hem_idx: (None or int) the index at which the hemispheres should be split. 
                    If not specified both hemispheres are analyzed at the same tiem. Default: None
    nonpar: (bool) do nonparameteric statistical significance testing as well
    """
    #> reorder and select valid parcels for X
    #  + convert them to np.ndarray
    X = X.loc[Y.index, Y.columns].values
    Y = Y.values
    #> make NaNs zero so NaN can be assigned to L-R edges
    X[np.isnan(X)] = 0 # make sure this is okay for all the matrices
    Y[np.isnan(Y)] = 0
    #> if L and R should be investigated separately
    # make interhemispheric pairs of the lower triangle
    # NaN so it could be then removed
    if split_hem_idx:
        X[split_hem_idx:, :split_hem_idx] = np.NaN
        Y[split_hem_idx:, :split_hem_idx] = np.NaN
    #> get the index for lower triangle
    tril_index = np.tril_indices_from(X, -1)
    x = X[tril_index]
    y = Y[tril_index]
    #> drop NaN (which assuming that there was no NaNs already
    #  in the matrix, corresponds to the interhemispheric pairs
    #  for the case of split hem correlations)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    #> remove 0s as the Y matrix is often zeroed out and 0
    #  has lost its meaning and there's a lot of zeros which
    #  have been actually the negative values
    mask = np.logical_and(x!=0, y!=0) 
    x = x[mask]
    y = y[mask]
    #> spearman correlation with 1000 permutation
    test_rho = scipy.stats.spearmanr(x, y).correlation
    p_val_par = scipy.stats.spearmanr(x, y).pvalue
    res_str = f"Correlation with {xlabel}\nRho: {test_rho}; Parametric p-value: {p_val_par}"
    # if nonpar:
    #     np.random.seed(921)
    #     null_rhos = np.array([])
    #     for _ in range(1000):
    #         surrogate = np.random.permutation(X) #> shuffles along the first dim and returns a copy
    #         surrogate = surrogate[tril_index]
    #         curr_perm_y = Y[tril_index]
    #         curr_perm_mask = np.logical_and(surrogate!=0, curr_perm_y!=0) # remove (often uninteresting) zeros from both matrices
    #         surrogate = surrogate[curr_perm_mask]
    #         curr_perm_y = curr_perm_y[curr_perm_mask]
    #         perm_rho = scipy.stats.spearmanr(surrogate, curr_perm_y).correlation
    #         null_rhos = np.append(null_rhos, perm_rho)
    #     p_val_nonpar = (np.abs(null_rhos) > np.abs(test_rho)).mean()
    #     res_str += f"Nonparametric p-value: {p_val_nonpar}"
    print(res_str)
    with open(prefix+'.txt', 'w') as res_file:
        res_file.write(res_str)
    #> plotting
    # jp = sns.jointplot(y=y, x=x, kind="kde", color="grey", height=4)
    jp = sns.jointplot(
        x = x, 
        y = y,
        kind = "hex", 
        color = "grey", 
        height = 4,
        marginal_kws = {'bins':35}, 
        joint_kws = {'gridsize':35},
        # xlim = (np.quantile(X[tril_index], 0.025), np.quantile(X[tril_index], 0.975)),
        # ylim = (np.quantile(Y[tril_index], 0.025), np.quantile(Y[tril_index], 0.975)),
    )
    ax = jp.ax_joint
    sns.regplot(X[tril_index], Y[tril_index], ax=ax, ci=None, scatter=False, color='C0', line_kws=dict(alpha=0.2))
    #> add rho on the figure
    text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
    text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.95
    ax.text(text_x, text_y, 
            f'rho = {test_rho:.2f}',
            color='black',
            size=16,
            multialignment='left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    jp.fig.tight_layout()
    jp.fig.savefig(
        prefix+'.png',
        dpi=192
        )
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


def correlate_matrices_node_wise(X, Y, prefix, parcellation_name, split_hem_idx=None):
    """
    Calculates the correlation between matrices `X` and `Y` at each row (node),
    projects the node-wise Spearman's rho values to the surface and saves and plots it. 
    Matrices are assumed to be square and symmetric.

    Paremeters
    ---------
    X, Y: (np.ndarray) n_parc x n_parc
    prefix: (str) prefix for the output filenames
    split_hem_idx: (None or int) the index at which the hemispheres should be split. 
                If not specified both hemispheres are analyzed at the same tiem. Default: None
    """
    #> reorder and select valid parcels for X
    #  + convert them to np.ndarray
    parcels = Y.index
    X = X.loc[Y.index, Y.columns].values
    Y = Y.values
    #> 
    node_rhos = np.empty(X.shape[0])
    node_pvals = np.empty(X.shape[0])
    for row_idx in range(X.shape[0]):
        row_x = X[row_idx, :]
        row_y = Y[row_idx, :]
        if split_hem_idx:
            if row_idx < split_hem_idx:
                row_x = row_x[:split_hem_idx]
                row_y = row_y[:split_hem_idx]
            else:
                row_x = row_x[split_hem_idx:]
                row_y = row_y[split_hem_idx:]
        #> calculate spearman correlation
        spearman_res = scipy.stats.spearmanr(row_x, row_y)
        node_rhos[row_idx] = spearman_res.correlation
        node_pvals[row_idx] = spearman_res.pvalue
    node_rhos = pd.Series(node_rhos, index=parcels).fillna(0) # replace NaN correlations with 0
    node_rhos_surface = helpers.deparcellate(node_rhos, parcellation_name)
    np.savez_compressed(prefix + '_nodewise_surface.npz', surface=node_rhos_surface)
    # helpers.plot_on_bigbrain_nl(
    #     node_rhos_surface,
    #     filename=prefix + '_nodewise_surface.png'
    # )
    _, node_pvals_fdr = statsmodels.stats.multitest.fdrcorrection(node_pvals)
    node_pvals_fdr = pd.Series(
        node_pvals_fdr,
        index=parcels)
    node_rhos_sig = node_rhos.copy()
    node_rhos_sig[node_pvals_fdr >= 0.05] = 0
    node_rhos_sig_surface = helpers.deparcellate(node_rhos_sig, parcellation_name)
    np.savez_compressed(prefix + '_nodewise_surface_sig.npz', surface=node_rhos_sig_surface)
    # helpers.plot_on_bigbrain_nl(
    #     node_rhos_sig_surface,
    #     filename=prefix + '_nodewise_surface_sig.png'
    # )

def correlate_laminar_similarity_matrix(matrix_file):
    #> get parcellation name based on matrix_file
    parcellation_name = re.match(r".*parc-([a-z|-|0-9]+)_*", matrix_file).groups()[0]
    #> load laminar similarity matrix (Y)
    Y_matrix = pd.read_csv(matrix_file, index_col='parcel')
    #> Load other similairty matrices of interest (X)
    X_matrices = {
        # GD matrix includes zeros for R-L edges
        'Geodesic distance': pd.read_csv(
            os.path.join(
                DATA_DIR, 'matrix',
                f'geodesic_parc-{parcellation_name}_approach-center-to-center.csv'
                ),
            index_col='parcel',
            ),
        'Curvature similarity': pd.read_csv(
            os.path.join(
                DATA_DIR, 'matrix',
                f'curvature_similarity_parc-{parcellation_name}.csv'
                ),
            index_col='parcel'
            ),
        'Microstructural profile covariance': load_mpc_matrix(parcellation_name),
    }
    #> with Schaefer parcellation add FC and SC as well
    #  TODO: calculate FC and SC for other parcellations as well
    if parcellation_name == 'schaefer400':
        X_matrices['Functional connectivity'], X_matrices['Structural connectivity'] = load_conn_matrices(matrix_file)
    #> If this is a density similarity matrix and has a thickness counterpart
    #  also add that to the X_matrices
    ylabel = 'Laminar thickness similarity'
    if 'thickness-density' in matrix_file:
        if 'matrix_input-density' in matrix_file:
            ylabel = 'Laminar density similarity'
            X_matrices['Laminar thickness similarity'] = pd.read_csv(
                matrix_file.replace('matrix_input-density.csv', 'matrix_input-thickness.csv'),
                index_col='parcel'
            )
    prefix = matrix_file.replace('.csv', '')
    for X_matrix_name, X_matrix in X_matrices.items():
        #> select and order X matrix parcels based on Y matrix
        X_matrix = X_matrix.loc[Y_matrix.index, Y_matrix.columns]
        #> determine the index of row/column where L hemisphere ends
        if X_matrix_name in ['Structural connectivity', 'Geodesic distance']:
            if ('excmask-adys' in matrix_file):
                parcels_adys = datasets.load_parcels_adys(parcellation_name)
                split_hem_idx = int(np.nansum(1-parcels_adys['L'].values)) # number of non-adys parcels in lh
            else:
                if parcellation_name == 'schaefer400':
                    split_hem_idx = 200 # TODO: this is specific for Schaefer parcellation
                elif parcellation_name == 'sjh':
                    split_hem_idx = 505
        else:
            split_hem_idx = None
        #> edge-wise correlations
        correlate_matrices_edge_wise(
            X = X_matrix,
            Y = Y_matrix,
            prefix = prefix + f'_correlation_{X_matrix_name.replace(" ","_").lower()}',
            xlabel = X_matrix_name,
            ylabel = ylabel,
            split_hem_idx = split_hem_idx
            )
        #> node-wise correlations
        if X_matrix_name in DO_NODE_WISE:
            print(f"Node-wise correlation with {X_matrix_name}")
            correlate_matrices_node_wise(
                X = X_matrix,
                Y = Y_matrix,
                prefix = prefix + f'_correlation_{X_matrix_name.replace(" ","_").lower()}',
                parcellation_name = parcellation_name,
                split_hem_idx = split_hem_idx
            )
        #> plotting the x matrix
        if X_matrix_name=='Geodesic distance':
            X_matrix[X_matrix==0] = np.NaN # so that interhemispheric pairs are transparent
        fig, ax = plt.subplots(figsize=(7,7))
        sns.heatmap(
            X_matrix,
            vmin=np.nanquantile(X_matrix.values.flatten(),0.025),
            vmax=np.nanquantile(X_matrix.values.flatten(),0.975),
            cbar=False,
            cmap=CMAPS[X_matrix_name],
            ax=ax)
        ax.axis('off')
        fig_outpath = os.path.join(
            os.path.dirname(prefix), 
            "matrix_"+ X_matrix_name.replace(" ","_").lower()
            )
        print(fig_outpath)
        fig.tight_layout()
        fig.savefig(fig_outpath+'.png', dpi=192)
        clfig = helpers.make_colorbar(
            vmin=np.nanquantile(X_matrix.values.flatten(),0.025),
            vmax=np.nanquantile(X_matrix.values.flatten(),0.975),
            cmap=CMAPS[X_matrix_name]
        )
        clfig.savefig(fig_outpath+'_clbar.png', dpi=192)

def associate_cortical_types(matrix_file):
    """
    Calculates within and between type similarity and plots a collapsed
    similarity matrix by cortical types. Excludes a-/dysgranular as indicated
    by the matrix filename

    Parameters
    ---------
    matrix_file: (str) path to matrix file    
    """
    matrix = pd.read_csv(matrix_file, index_col='parcel')
    matrix.columns = matrix.index # for sjh parcellation index is loaded as int but columns as str
    #> get parcellation name based on matrix_file
    parcellation_name = re.match(r".*parc-([a-z|-|0-9]+)_*", matrix_file).groups()[0]
    #> load cortical types and make it match matrix index
    parcellated_cortical_types = datasets.load_cortical_types(parcellation_name)
    parcellated_cortical_types = parcellated_cortical_types.loc[matrix.index]
    #> list the excluded types based on matrix path 
    if ('excmask' in matrix_file):
        included_types = ['EU1', 'EU2', 'EU3', 'KO']
        excluded_types = ['ALO', 'AG', 'DG']
        n_types = 4
    else:
        included_types = ['AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO']
        excluded_types = ['ALO']
        n_types = 6
    matrix = matrix.loc[
        ~parcellated_cortical_types.isin(excluded_types), 
        ~parcellated_cortical_types.isin(excluded_types)
    ]
    parcellated_cortical_types = (parcellated_cortical_types
        .loc[~parcellated_cortical_types.isin(excluded_types)]
        .cat.remove_unused_categories())
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
    if ('thickness' in matrix_file) or ('volume' in matrix_file):
        cmap = sns.color_palette("RdBu_r", as_cmap=True)
    elif 'density' in matrix_file:
        cmap = sns.color_palette("viridis", as_cmap=True)
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(
        mean_matrix_by_cortical_type,
        cmap=cmap,
        cbar=False,
        ax=ax)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(matrix_file.replace('.csv', '_collapsed_cortical_types'))
    helpers.make_colorbar(
        vmin=mean_matrix_by_cortical_type.min().min(),
        vmax=mean_matrix_by_cortical_type.max().min(),
        cmap=cmap
    ).savefig(matrix_file.replace('.csv', '_collapsed_cortical_types_clbar'))
    #> quantify intra and intertype similarity
    intra_intertype = pd.DataFrame(
        np.zeros((n_types, 2)),
        columns=['intra', 'inter'],
        index=included_types)
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
    with open(matrix_file.replace('.csv', '_intra_intertype_diff.txt'), 'w') as outfile:
        outfile.write(str(intra_intertype))


matrix_files = glob.glob(os.path.join(DATA_DIR, 'result', '*volume*', 'matrix*.csv'))
for matrix_file in matrix_files:
    print("Matrix:", matrix_file)
    matrix = pd.read_csv(matrix_file, index_col='parcel')
    if matrix.shape[0] != matrix.shape[1]:
        print("\tNot square")
        continue
    else:
        associate_cortical_types(matrix_file)
        correlate_laminar_similarity_matrix(matrix_file)