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
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.surface
import helpers


#> specify the data dir and create gradients and matrices subfolders
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')


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
    parcellated_gradients = helpers.parcellate(gradient_maps.T, 'sjh')
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

#> sample gradient file path
gradient_files = glob.glob(os.path.join(DATA_DIR, 'gradient', '*input-thickness_*_excmask_*_surface.npz'))
print(gradient_files)
for gradient_file in gradient_files:
    out = associate_cortical_types(gradient_file)    
