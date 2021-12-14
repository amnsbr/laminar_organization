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
import helpers
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import enigmatoolbox.datasets
import cmcrameri.cm


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
    reordered_fc_ctx, reordered_sc_ctx: (np.ndarray) (400, 400) reordered FC and SC matrices
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
    reordered_sc_ctx.iloc[:200, 200:] = 0
    reordered_sc_ctx.iloc[200:, :200] = 0
    return reordered_fc_ctx, reordered_sc_ctx

def correlate_matrices_edge_wise(X, Y, prefix, xlabel, ylabel, nonpar=False):
    """
    Calculates and plots the correlation between the edges of two matrices
    `X` and `Y` which are assumed to be square and symmetric. The correlation
    is calculated for lower triangle (excluding the diagonal)

    Paremeters
    ---------
    X, Y: (pd.DataFrame) n_parc x n_parc
    prefix: (str) prefix for the output filename
    xlabel, ylabel: (str)
    nonpar: (bool) do nonparameteric statistical significance testing as well
    """
    #> reorder and select valid parcels for X
    #  + convert them to np.ndarray
    X = X.loc[Y.index, Y.columns].values
    Y = Y.values
    #> get the index for lower triangle
    tril_index = np.tril_indices_from(X, -1)
    x = X[tril_index]
    y = Y[tril_index]
    #> remove (often uninteresting) zeros from both matrices
    #  e.g. zeroed out similarity values, or non-existing GD (R to L)
    #  TODO: make sure we are not removing anything interesting!
    mask = np.logical_and(x!=0, y!=0) 
    x = x[mask]
    y = y[mask]
    #> spearman correlation with 1000 permutation
    np.random.seed(921)
    test_rho = scipy.stats.spearmanr(x, y).correlation
    p_val_par = scipy.stats.spearmanr(x, y).pvalue
    res_str = f"Correlation with {xlabel}\nRho: {test_rho}; Parametric p-value: {p_val_par}"
    if nonpar:
        null_rhos = np.array([])
        for _ in range(1000):
            surrogate = np.random.permutation(X) #> shuffles along the first dim and returns a copy
            surrogate = surrogate[tril_index]
            curr_perm_y = Y[tril_index]
            curr_perm_mask = np.logical_and(surrogate!=0, curr_perm_y!=0) # remove (often uninteresting) zeros from both matrices
            surrogate = surrogate[curr_perm_mask]
            curr_perm_y = curr_perm_y[curr_perm_mask]
            perm_rho = scipy.stats.spearmanr(surrogate, curr_perm_y).correlation
            null_rhos = np.append(null_rhos, perm_rho)
        p_val_nonpar = (np.abs(null_rhos) > np.abs(test_rho)).mean()
        res_str += f"Nonparametric p-value: {p_val_nonpar}"
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
        marginal_kws = {'bins':50}, 
        joint_kws = {'gridsize':50},
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


def correlate_matrices_node_wise(X, Y, prefix, parcellation_name):
    """
    Calculates the correlation between matrices `X` and `Y` at each row (node),
    projects the node-wise Spearman's rho values to the surface and saves and plots it. 
    Matrices are assumed to be square and symmetric.

    Paremeters
    ---------
    X, Y: (np.ndarray) n_parc x n_parc
    prefix: (str) prefix for the output filenames
    """
    #> reorder and select valid parcels for X
    #  + convert them to np.ndarray
    parcels = Y.index
    X = X.loc[Y.index, Y.columns].values
    Y = Y.values
    #> 
    node_rhos = np.empty(X.shape[0])
    for row_idx in range(X.shape[0]):
        #> remove zero values (TODO make sure this is valid with all matrices)
        row_x = X[row_idx, :]
        row_y = Y[row_idx, :]
        mask = np.logical_and(row_x!=0, row_y!=0)
        row_x = row_x[mask]
        row_y = row_y[mask]
        #> calculate spearman correlation
        node_rhos[row_idx] = scipy.stats.spearmanr(row_x, row_y).correlation
    node_rhos = pd.Series(node_rhos, index=parcels)
    node_rhos_surface = helpers.deparcellate(node_rhos, parcellation_name)
    np.savez_compressed(prefix + '_nodewise_surface.npz', surface=node_rhos_surface)
    helpers.plot_on_bigbrain_nl(
        node_rhos_surface,
        filename=prefix + '_nodewise_surface.png'
    )

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
        X_matrix = X_matrix.loc[Y_matrix.index, Y_matrix.index]
        #> edge-wise correlations
        correlate_matrices_edge_wise(
            X = X_matrix,
            Y = Y_matrix,
            prefix = prefix + f'_correlation_{X_matrix_name.replace(" ","_").lower()}',
            xlabel = X_matrix_name,
            ylabel = ylabel
            )
        #> node-wise correlations
        if X_matrix_name in DO_NODE_WISE:
            print(f"Node-wise correlation with {X_matrix_name}")
            correlate_matrices_node_wise(
                X = X_matrix,
                Y = Y_matrix,
                prefix = prefix + f'_correlation_{X_matrix_name.replace(" ","_").lower()}',
                parcellation_name = parcellation_name
            )
        #> plotting the x matrix
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


matrix_files = glob.glob(os.path.join(DATA_DIR, 'result', '*', 'matrix*.csv'))
for matrix_file in matrix_files:
    print("Matrix:", matrix_file)
    matrix = pd.read_csv(matrix_file, index_col='parcel')
    if matrix.shape[0] != matrix.shape[1]:
        print("\tNot square")
        continue
    else:
        correlate_laminar_similarity_matrix(matrix_file)