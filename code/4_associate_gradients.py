"""
Associate gradients with:
1. Cortical types
2. BigBrain histological gradients
3. BigBrain density moments
3. Laminar thicknesses and properties
"""
import os
import glob
import itertools
import gc
import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.surface
import helpers
import brainspace.null_models
import brainspace.mesh



#> specify the data dir and create gradients and matrices subfolders
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')
SRC_DIR = os.path.join(cwd, '..', 'src')
SPIN_BATCHES_DIR = os.path.join(SRC_DIR, 'spin_batches')
os.makedirs(SPIN_BATCHES_DIR, exist_ok=True)


def load_cortical_types_map(save=True):
    #> load the economo map and concatenate left and right hemispheres
    economo_maps = {}
    for hem in ['L', 'R']:
        economo_maps[hem] = nilearn.surface.load_surf_data(
            os.path.join(
                DATA_DIR, 'parcellation',
                f'tpl-bigbrain_hemi-{hem}_desc-economo_parcellation.label.gii'
                )
            )
    economo_map = np.concatenate([economo_maps['L'], economo_maps['R']])
    #> load the cortical types for each economo parcel
    economo_cortical_types = pd.read_csv(
        os.path.join(
            DATA_DIR, 'parcellated_surface',
            'economo_cortical_types.csv'
            )
        )
    economo_cortical_types.columns=['Label', 'Cortical Type']
    #> create the cortical types surface map
    cortical_types_map = economo_cortical_types.loc[economo_map, 'Cortical Type'].astype('category').reset_index(drop=True)
    cortical_types_map = cortical_types_map.cat.reorder_categories(['ALO', 'AG', 'DG', 'EU1', 'EU2', 'EU3', 'KO'])
    #> save the map
    if save:
        # TODO: split hemispheres
        np.save(
            os.path.join(
                DATA_DIR, 'parcellation',
                f'tpl-bigbrain_desc-cortical_types_parcellation.npy'
            ), cortical_types_map.cat.codes.values)
    return cortical_types_map

def associate_cortical_types(gradient_file, n_gradients=3):
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> load cortical types map
    cortical_types_map = load_cortical_types_map()
    #> parcellate the gradient maps
    parcellated_gradients = helpers.parcellate(gradient_maps, 'sjh')
    #> parcellate cortical types (using the most frequent type)
    parcellation_map = helpers.load_parcellation_map('sjh', concatenate=True)
    parcellated_cortical_types = (
        #>> create a dataframe of surface map including both cortical type and parcel index
        pd.DataFrame({'Cortical Type': cortical_types_map.reset_index(drop=True), 'Parcel': pd.Series(parcellation_map)})
        #>> group by parcel
        .groupby('Parcel')
        #>> find the cortical types with the highest count
        ['Cortical Type'].value_counts(sort=True).unstack().idxmax(axis=1)
        )
    #> create a df combining the gradient values and cortical types at each parcel
    gradients_cortical_types = parcellated_gradients.copy()
    gradients_cortical_types['Cortical Type'] = parcellated_cortical_types.astype('category')
    #> exclude some types
    if ('excmask' in gradient_file):
        excluded_types = ['ALO', 'AG', 'DG']
    else:
        excluded_types = ['ALO']
    gradients_cortical_types = gradients_cortical_types[~gradients_cortical_types['Cortical Type'].isin(excluded_types)]
    gradients_cortical_types['Cortical Type'] = gradients_cortical_types['Cortical Type'].cat.remove_unused_categories()
    gradients_cortical_types.columns = [f'G{gradient_num}' for gradient_num in range(1, gradient_maps.shape[1]+1)] + ['Cortical Type']

    #> investigate the association of gradient values and cortical types (visually and statistically)
    anova_res_str = "ANOVA Results\n--------\n"
    for gradient_num in range(1, n_gradients+1):
        fig, ax = plt.subplots(figsize=(4, 4))
        #> boxplot
        ax = sns.violinplot(
            data=gradients_cortical_types, 
            y=f'G{gradient_num}',
            x='Cortical Type',
            palette='Set3',
            bw=.5, cut=1, linewidth=1,
            ax=ax
            )
        #> aesthetics
        plt.setp(ax.collections, alpha=.6)
        sns.despine(ax=ax, offset=10, trim=True)
        fig.tight_layout()
        fig.savefig(
            gradient_file.replace('surface.npz', f'cortical_type_G{gradient_num}.png'),
            dpi=192
            )
        #> colorbar with the correct vmin and vmax
        # clbar_fig = helpers.make_colorbar(ax.get_yticks()[0], ax.get_yticks()[-1], figsize=(4, 4))
        # clbar_fig.tight_layout()
        # clbar_fig.savefig(os.path.join(PROJECT_ROOT, 'figures', f'partial_corr_w_adysgranular_cortical_type_violinplot_G{gradient_num}_colorbar.png'), dpi=192);
        #> ANOVA
        F, p_val = (scipy.stats.f_oneway(*[
                                    cortical_type_data[1][f'G{gradient_num}'].dropna().values \
                                    for cortical_type_data in gradients_cortical_types.groupby('Cortical Type')
                                    ]))
        anova_res_str += f'----\nGradient {gradient_num}: F statistic {F}, pvalue {p_val}\n'
        if p_val < 0.05:
            alpha = 0.05 / len(list(itertools.combinations(gradients_cortical_types['Cortical Type'].cat.categories, 2))) #bonferroni correction
            anova_res_str += f"\tPost-hocs passing alpha of {alpha}:\n"
            for type1, type2 in itertools.combinations(gradients_cortical_types['Cortical Type'].cat.categories, 2):
                t_statistic, t_p = scipy.stats.ttest_ind(
                    gradients_cortical_types.loc[gradients_cortical_types['Cortical Type']==type1, f'G{gradient_num}'].dropna(),
                    gradients_cortical_types.loc[gradients_cortical_types['Cortical Type']==type2, f'G{gradient_num}'].dropna(),
                )
                if t_p < alpha:
                    anova_res_str += f"\t\t{type1} vs {type2}: T {t_statistic}, p {t_p}\n"
    print(anova_res_str)
    with open(gradient_file.replace('surface.npz', 'cortical_type_anova.txt'), 'w') as anova_res_file:
        anova_res_file.write(anova_res_str)

def create_spin_permutations(surface_data, n_perm, batch_prefix, batch_size=20):
    """
    Creates spin permutations of the input map and saves them in batches on 'src' folder

    surface_data: (np.ndarray) n_vert x n_features
    n_perm: (int) total number of permutations
    batch_prefix: (str) batches filename prefix
    batch_size: (int) number of permutations per batch
    """
    if os.path.exists(os.path.join(SPIN_BATCHES_DIR, f'{batch_prefix}_batch0.npz')):
        print("Spin permutation batches already exist")
    #> read the bigbrain surface giftii files as a mesh that can be used by spin_permutations function
    lh_surf = brainspace.mesh.mesh_io.read_surface(os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-mid.surf.gii'))
    rh_surf = brainspace.mesh.mesh_io.read_surface(os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-mid.surf.gii'))
    #> create permutations of the first five gradients with preserved spatial-autocorrelation using spin_permutations
    #  doing it in batches because colab shuts down sometimes! Also all permutations in one giant matrix requires too much
    #  memory (and colab crashes). So I will keep these in batches and do the analyses on batches to save memory.
    n_batch = n_perm // batch_size
    for batch in range(n_batch):
        print('\t\tBatch', batch)
        batch_lh_rand, batch_rh_rand = brainspace.null_models.spin.spin_permutations(
            spheres = {'lh': lh_surf,
                    'rh': rh_surf},
            data = {'lh': surface_data[:surface_data.shape[0]//2],
                    'rh': surface_data[surface_data.shape[0]//2:]},
            n_rep = batch_size,
            random_state = 9*batch, # it's important for random states to be different across batche
        )
        np.savez_compressed(os.path.join(SPIN_BATCHES_DIR, f'{batch_prefix}_batch{batch}.npz'), lh=batch_lh_rand, rh=batch_rh_rand)

def spin_correlation_test(surface_data_to_spin, surface_data_target, parcellation_name, batch_prefix):
    """
    Performs spin test on the correlation between x and y after parcellation, 
    where x is already spin permutated and the permutations are stored in batches_dir

    surface_data_to_spin: (np.ndarray) n_vert * n_features (both hemispheres) [spinned one]
    surface_data_target: (np.ndarray) n_vert * n_features (both hemispheres)
    parcellation_name: (str) parcellation name. must exist in data/parcellations in bigbrain space
    batch_prefix: (str) batches filename prefix
    """
    #> parcellate data
    parcellated_surface_data_to_spin = helpers.parcellate(surface_data_to_spin, parcellation_name)
    parcellated_surface_data_target = helpers.parcellate(surface_data_target, parcellation_name)
    #> calculate test correlation coefficient between all gradients and all other surface maps
    test_r = (
        pd.concat([parcellated_surface_data_to_spin, parcellated_surface_data_target], axis=1)
        .corr() # this will calculate the correlation coefficient between all the gradients and other surface maps
        .iloc[:parcellated_surface_data_to_spin.shape[1], -parcellated_surface_data_target.shape[1]:] # select only the correlations we are interested in
        .T.values[np.newaxis, :] # convert it to shape (1, n_features_surface_data_target, n_features_surface_data_to_spin)
    )
    null_distribution = test_r.copy() # will have the shape (n_perms, n_features_surface_data_target, n_features_surface_data_to_spin)
    for batch_file in sorted(glob.glob(os.path.join(SPIN_BATCHES_DIR, f'{batch_prefix}_batch*.npz'))):
        print("\t\tBatch", batch_file)
        #> load the 20-spin batch of spin permutated maps and concatenate left and right hemispheres
        #  only select the first n_gradients
        batch_perms = np.load(batch_file)
        batch_lh_rand = batch_perms['lh'][:, :, :surface_data_to_spin.shape[1]]
        batch_rh_rand = batch_perms['rh'][:, :, :surface_data_to_spin.shape[1]]
        batch_both_rand = np.concatenate([batch_lh_rand, batch_rh_rand], axis=1)
        for perm_idx in range(batch_rh_rand.shape[0]):
            #> parcellate the spin permutated gradient map
            parcellated_rand = helpers.parcellate(batch_both_rand[perm_idx, :], parcellation_name)
            #> calculate null correlation coefficient between all gradients and all other surface maps
            null_r = (
                pd.concat([parcellated_rand, parcellated_surface_data_target], axis=1)
                .corr() # this will calculate the correlation coefficient between all the gradients and other surface maps
                .iloc[:parcellated_rand.shape[1], -parcellated_surface_data_target.shape[1]:] # select only the correlations we are interested in
                .T.values[np.newaxis, :] # convert it to shape (1, n_features_surface_data_target, n_features_surface_data_to_spin)
            )
            #> add this to the null distribution
            null_distribution = np.concatenate([null_distribution, null_r], axis=0)
            #> free up memory
            gc.collect()
    #> remove the test_r from null_distribution
    null_distribution = null_distribution[1:, :, :]
    #> calculate p value
    p_val = (np.abs(null_distribution) >= np.abs(test_r)).mean(axis=0)
    #> reduce unnecessary dimension of test_r
    test_r = test_r[0, :, :]
    return test_r, p_val, null_distribution

def correlate_hist_gradients(gradient_file, n_laminar_gradients, n_perm):
    print(f"Investigating correlation with Hist MPC gradients: {gradient_file}")
    #> load hist mpc gradients in n_vert * n_gradients shape
    hist_gradients = {}
    for hist_gradient_num in range(1, 3):
        hist_gradients[hist_gradient_num] = {}
        for hem in ['L', 'R']:
            hist_gradients[hist_gradient_num][hem] = np.loadtxt(
                os.path.join(
                    DATA_DIR, 'gradient',
                    f'tpl-bigbrain_hemi-{hem}_desc-Hist_G{hist_gradient_num}.txt'
                )
            )
        hist_gradients[hist_gradient_num] = np.concatenate([
            hist_gradients[hist_gradient_num]['L'], 
            hist_gradients[hist_gradient_num]['R']
            ])
    hist_gradients = np.vstack([
        hist_gradients[1],
        hist_gradients[2],
    ]).T
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> create spin permtations of hist gradients
    print(f"\tCreating {n_perm} spin permutations for Hist gradients")
    create_spin_permutations(hist_gradients, n_perm, 'hist_gradients')
    print(f"\tCalculating correlations with spin test")
    coefs, pvals, coefs_null_dist =  spin_correlation_test(
        surface_data_to_spin = hist_gradients, 
        surface_data_target = gradient_maps[:, :n_laminar_gradients], 
        parcellation_name='sjh',
        batch_prefix='hist_gradients')
    coefs = pd.DataFrame(coefs)
    pvals = pd.DataFrame(pvals)
    coefs.columns = pvals.columns = ['Hist-G1', 'Hist-G2']
    coefs.index = pvals.index = [f'Laminar-G{gradient_num}' for gradient_num in range(1, n_laminar_gradients+1)]
    #> save the results as txt
    association_res_str = f"Association with Hist gradients (spin test)\n--------\nCorrelation coefficients:{coefs}\nP-values:{pvals}\n"
    print(association_res_str)
    with open(gradient_file.replace('surface.npz', f'corr_HistG.png'), 'w') as association_res_file:
        association_res_file.write(association_res_str)
    #> regression plots
    parcellated_gradients = helpers.parcellate(gradient_maps, 'sjh')
    parcellated_hist_gradients = helpers.parcellate(hist_gradients, 'sjh')
    for gradient_num in range(1, n_laminar_gradients+1):
        fig, ax = plt.subplots(figsize=(4, 4))
        #> regression plot with hist G1 in blue
        sns.regplot(x=parcellated_gradients.iloc[:,gradient_num-1], y=parcellated_hist_gradients.iloc[:, 0], color='C0', scatter_kws={'alpha':0.2, 's':5}, ax=ax)
        #> regression plot with hist G2 in green
        sns.regplot(x=parcellated_gradients.iloc[:,gradient_num-1], y=parcellated_hist_gradients.iloc[:, 1], color='C2', scatter_kws={'alpha':0.2, 's':5}, ax=ax)
        sns.despine(offset=10, trim=True, ax=ax)
        ax.set_xlabel(f'G{gradient_num}')
        ax.set_ylabel('')
        #> add correlation coefficients and p vals on the figure
        text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
        text_y1 = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.10
        text_y2 = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
        ax.text(text_x, text_y1, 
                f'r = {coefs.iloc[gradient_num-1, 0]:.2f}; $\mathregular{{p_{{spin}}}}$ = {pvals.iloc[gradient_num-1, 0]:.2f}',
                color='C0',
                size=8,
                multialignment='left')
        ax.text(text_x, text_y2, 
                f'r = {coefs.iloc[gradient_num-1, 1]:.2f}; $\mathregular{{p_{{spin}}}}$ = {pvals.iloc[gradient_num-1, 1]:.2f}',
                color='C2',
                size=8,
                multialignment='left')
        fig.tight_layout()
        fig.savefig(
            gradient_file.replace('surface.npz', f'corr_HistG_G{gradient_num}.png'),
            dpi=192
            )

#> sample gradient file path
gradient_files = glob.glob(os.path.join(DATA_DIR, 'gradient', '*input-thickness_*_surface.npz'))
# gradient_files = glob.glob(os.path.join(DATA_DIR, 'gradient', '*_surface.npz'))
print(gradient_files)
for gradient_file in gradient_files[:1]:
    # associate_cortical_types(gradient_file)
    correlate_hist_gradients(gradient_file, n_laminar_gradients=3, n_perm=100)
