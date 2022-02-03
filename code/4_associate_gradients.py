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
import re
import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.surface
import nibabel
import brainspace.null_models
import brainspace.mesh
import brainsmash.mapgen
import ptitprince
from statsmodels import formula # for RainCloud plots
import statsmodels.api as sm
import statsmodels.formula.api as smf

import helpers
import datasets


#> specify the data dir and create gradients and matrices subfolders
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')
SRC_DIR = os.path.join(cwd, '..', 'src')

N_HEM_VERTICES = np.loadtxt(
        os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )
        ).size

#### Associations ####
def associate_cortical_types(gradient_file, n_gradients=3):
    """
    Calculates and plots the association of `gradient_file` first `n_gradients` with
    the cortical types using ANOVA. The ANOVA results are stored in a txt file with
    _cortical_type_anova.txt suffix

    Parameters
    ---------
    gradient_file: (str) input gradient file (.npz format) including an array 'surface' 
                         with the shape n_vert x total_n_gradients
    n_gradients: (int) number of gradients to associate with cortical types
    """
    #> get parcellation name
    parcellation_name = re.match(r".*parc-([a-z|-|0-9]+)_*", gradient_file).groups()[0]
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> parcellate the gradient maps
    parcellated_gradients = helpers.parcellate(gradient_maps, parcellation_name)
    #> load parcellated cortical types map
    parcellated_cortical_types = datasets.load_cortical_types(parcellation_name)
    #> create a df combining the gradient values and cortical types at each parcel
    gradients_cortical_types = parcellated_gradients.copy()
    gradients_cortical_types['Cortical Type'] = parcellated_cortical_types.astype('category')
    #> specify type colors
    type_colors = sns.color_palette("RdYlGn_r", 6)
    #> exclude some types
    if ('excmask' in gradient_file):
        excluded_types = ['ALO', 'AG', 'DG']
        type_colors = type_colors[2:]
    else:
        excluded_types = ['ALO']
    gradients_cortical_types = gradients_cortical_types[~gradients_cortical_types['Cortical Type'].isin(excluded_types)]
    gradients_cortical_types['Cortical Type'] = gradients_cortical_types['Cortical Type'].cat.remove_unused_categories()
    gradients_cortical_types.columns = [f'G{gradient_num}' for gradient_num in range(1, gradient_maps.shape[1]+1)] + ['Cortical Type']

    #> investigate the association of gradient values and cortical types (visually and statistically)
    anova_res_str = "ANOVA Results\n--------\n"
    for gradient_num in range(1, n_gradients+1):
        # fig, ax = plt.subplots(figsize=(4, 4))
        # #> violinplot
        # ax = sns.violinplot(
        #     data=gradients_cortical_types, 
        #     y=f'G{gradient_num}',
        #     x='Cortical Type',
        #     palette=type_colors,
        #     bw=.5, cut=1, linewidth=1,
        #     ax=ax
        #     )
        # #>> aesthetics
        # plt.setp(ax.collections, alpha=.6)
        # sns.despine(ax=ax, offset=10, trim=True)
        #> raincloud plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax=ptitprince.RainCloud(
            data=gradients_cortical_types, 
            y=f'G{gradient_num}',
            x='Cortical Type',
            palette=type_colors,
            bw = 0.2, width_viol = 1, 
            orient = 'h', move = 0.2, alpha = 0.4,
            ax=ax)
        sns.despine(ax=ax, offset=10, trim=True)
        fig.tight_layout()
        fig.savefig(
            gradient_file.replace('gradients_surface.npz', f'cortical_type_G{gradient_num}.png'),
            dpi=192
            )
        #>> colorbar with the correct vmin and vmax
        clbar_fig = helpers.make_colorbar(ax.get_yticks()[0], ax.get_yticks()[-1], figsize=(4, 4))
        clbar_fig.tight_layout()
        clbar_fig.savefig(
            gradient_file.replace('gradients_surface.npz', f'cortical_type_G{gradient_num}_clbar.png'), 
            dpi=192)
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
    with open(gradient_file.replace('gradients_surface.npz', 'cortical_type_anova.txt'), 'w') as anova_res_file:
        anova_res_file.write(anova_res_str)

def associate_yeo_networks(gradient_file, n_gradients=3):
    """
    Calculates and plots the association of `gradient_file` first `n_gradients` with
    the yeo networks using ANOVA. The ANOVA results are stored in a txt file with
    _yeo_networks_anova.txt suffix.
    Two plots are created: violin plot and stacked bar plot

    Parameters
    ---------
    gradient_file: (str) input gradient file (.npz format) including an array 'surface' 
                         with the shape n_vert x total_n_gradients
    n_gradients: (int) number of gradients to associate with cortical types
    """
    # TODO: this is very similar to associate_cortical_types => merge them
    #> get parcellation name
    parcellation_name = re.match(r".*parc-([a-z|-|0-9]+)_*", gradient_file).groups()[0]
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> load yeo networks map and colors
    yeo_giftis = {}
    for hem in ['L', 'R']:
        yeo_giftis[hem] = nibabel.load(
            os.path.join(
                DATA_DIR, 'parcellation',
                f'tpl-bigbrain_hemi-{hem}_desc-Yeo2011_7Networks_N1000.label.gii'
            )
        )
    yeo_map = np.concatenate([
        yeo_giftis['L'].darrays[0].data, 
        yeo_giftis['R'].darrays[0].data
        ])
    yeo_colors = [l.rgba[:-1] for l in yeo_giftis['L'].labeltable.labels[1:]]
    yeo_names = [
        'Visual', 'Somatomotor', 'Dorsal attention', 
        'Ventral attention', 'Limbic', 'Frontoparietal', 'Default'
        ]
    #> create a df combining the gradient values and yeo_networks
    gradient_map_yeo = pd.DataFrame(
        gradient_maps[:, :n_gradients],
        columns=[f'G{g_n}' for g_n in range(1, n_gradients+1)]
        )
    gradient_map_yeo['Yeo Network'] = yeo_map
    #> remove NaNs and Yeo 0 (midline)
    gradient_map_yeo = gradient_map_yeo[gradient_map_yeo['Yeo Network']!=0].dropna()
    gradient_map_yeo['Yeo Network'] = (gradient_map_yeo['Yeo Network']
                                    .astype('category')
                                    .cat.remove_unused_categories()
                                    .cat.rename_categories(yeo_names))
    #> investigate the association of gradient values and Yeo networks (visually and statistically)
    anova_res_str = "ANOVA Results\n--------\n"
    for g_n in range(1, n_gradients+1):
        #> 1) plot networks frequency in each gradient bin
        #>> assign each vertex to one of the 10 bins
        _, bin_edges = np.histogram(gradient_map_yeo.loc[:,f'G{g_n}'], bins=10)
        gradient_map_yeo[f'G{g_n}_bin'] = np.digitize(gradient_map_yeo.loc[:,f'G{g_n}'], bin_edges[:-1])
        #>> calculate ratio of yeo networks in each bin
        gradient_bins_yeo_counts = (gradient_map_yeo
                                    .groupby([f'G{g_n}_bin','Yeo Network'])
                                    .size().unstack(fill_value=0))
        gradient_bins_yeo_freq = gradient_bins_yeo_counts.divide(gradient_bins_yeo_counts.sum(axis=1), axis=0)
        #>> plot stacked bars at each bin showing freq of the Yeo networks
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x = gradient_bins_yeo_freq.index,
            height = gradient_bins_yeo_freq.iloc[:, 0],
            width = 0.95,
            color=yeo_colors[0],
            label=yeo_names[0]
            )
        for yeo_idx in range(1, 7):
            ax.bar(
                x = gradient_bins_yeo_freq.index,
                height = gradient_bins_yeo_freq.iloc[:, yeo_idx],
                width = 0.95,
                bottom = gradient_bins_yeo_freq.cumsum(axis=1).iloc[:, yeo_idx-1],
                color=yeo_colors[yeo_idx],
                label=yeo_names[yeo_idx]
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'G{g_n} bins')
        ax.set_ylabel('Proportion of networks')
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(
            gradient_file.replace('gradients_surface.npz', f'yeo_network_stacked_G{g_n}.png'),
            dpi=192
            )
        #> 2) violinplot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax = sns.violinplot(
            data=gradient_map_yeo, 
            y=f'G{g_n}',
            x='Yeo Network',
            palette=yeo_colors,
            bw=.5, cut=1, linewidth=1,
            ax=ax
            )
        plt.setp(ax.collections, alpha=.6)
        sns.despine(ax=ax, offset=10, trim=True)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
        fig.tight_layout()
        fig.savefig(
            gradient_file.replace('gradients_surface.npz', f'yeo_network_violin_G{g_n}.png'),
            dpi=192
            )
        #> 3) ANOVA [TODO: all tests are bound to be sig at the vertex level; fix it!]
        F_stat, p_val = (scipy.stats.f_oneway(*[
                                    yeo_network_data[1][f'G{g_n}'].dropna().values \
                                    for yeo_network_data in gradient_map_yeo.groupby('Yeo Network')
                                    ]))
        anova_res_str += f'----\nGradient {g_n}: F statistic {F_stat}, pvalue {p_val}\n'
        if p_val < 0.05:
            alpha = 0.05 / len(list(itertools.combinations(gradient_map_yeo['Yeo Network'].cat.categories, 2))) #bonferroni correction
            anova_res_str += f"\tPost-hocs passing alpha of {alpha}:\n"
            for network1, network2 in itertools.combinations(gradient_map_yeo['Yeo Network'].cat.categories, 2):
                t_statistic, t_p = scipy.stats.ttest_ind(
                    gradient_map_yeo.loc[gradient_map_yeo['Yeo Network']==network1, f'G{g_n}'].dropna(),
                    gradient_map_yeo.loc[gradient_map_yeo['Yeo Network']==network2, f'G{g_n}'].dropna(),
                )
                if t_p < alpha:
                    anova_res_str += f"\t\t{network1} - {network2}: T {t_statistic}, p {t_p}\n"
    print(anova_res_str)
    with open(gradient_file.replace('gradients_surface.npz', 'yeo_networks_anova.txt'), 'w') as anova_res_file:
        anova_res_file.write(anova_res_str)

def correlate_hist_gradients(gradient_file, n_laminar_gradients, n_perm):
    """
    Calculates the correlation between Hist MPC gradients 1 & 2 with laminar gradient
    `gradient_file` (its first `n_laminar_gradients`). Corrects for spatial
    autocorrelation by doing spin test with `n_perm` permutations.

    Parameters
    ---------
    gradient_file: (str) input gradient file (.npz format) including an array 'surface' 
                         with the shape n_vert x total_n_gradients
    n_laminar_gradients: (int) number of gradients to associate with cortical types
    n_perm: (int) number of spin permutations
    """
    #> get parcellation name
    parcellation_name = re.match(r".*parc-([a-z|-|0-9]+)_*", gradient_file).groups()[0]
    print(f"Investigating correlation with Hist MPC gradients: {gradient_file}")
    #> load hist mpc gradients in n_vert * n_gradients shape
    hist_gradients = datasets.load_hist_mpc_gradients()
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> spin test
    print(f"\tCalculating correlations with spin test")
    coefs, pvals, coefs_null_dist =  helpers.spin_test(
        surface_data_to_spin = hist_gradients, 
        surface_data_target = gradient_maps[:, :n_laminar_gradients],)
    #> save null distribution for future reference
    np.savez_compressed(
        gradient_file.replace('gradients_surface.npz', 'correlation_HistG_null.npz'),
        coefs_null_dist=coefs_null_dist
    )
    #> clean and save the results as txt
    coefs = pd.DataFrame(coefs)
    pvals = pd.DataFrame(pvals)
    coefs.columns = pvals.columns = ['Hist-G1', 'Hist-G2']
    coefs.index = pvals.index = [f'Laminar-G{gradient_num}' for gradient_num in range(1, n_laminar_gradients+1)]
    association_res_str = f"Association with Hist gradients (spin test)\n--------\nCorrelation coefficients:{coefs}\nP-values:{pvals}\n"
    print(association_res_str)
    with open(gradient_file.replace('gradients_surface.npz', f'correlation_HistG.txt'), 'w') as association_res_file:
        association_res_file.write(association_res_str)
    #> regression plots
    parcellated_gradients = helpers.parcellate(gradient_maps, parcellation_name)
    parcellated_hist_gradients = helpers.parcellate(hist_gradients, parcellation_name)
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
            gradient_file.replace('gradients_surface.npz', f'correlation_HistG_G{gradient_num}.png'),
            dpi=192
            )

def correlate_laminar_properties_and_moments(gradient_file, n_laminar_gradients, n_perm):
    """
    Calculates the correlation between laminar properties + density moments 
    with laminar gradient `gradient_file` (its first `n_laminar_gradients`). 
    Corrects for spatial autocorrelation by doing spin test with `n_perm` permutations.

    Parameters
    ---------
    gradient_file: (str) input gradient file (.npz format) including an array 'surface' 
                         with the shape n_vert x total_n_gradients
    n_laminar_gradients: (int) number of gradients to associate with cortical types
    n_perm: (int) number of spin permutations
    """
    #> load parcellation name
    parcellation_name = re.match(r".*parc-([a-z|-|0-9]+)_*", gradient_file).groups()[0]
    print(f"Investigating correlation with laminar properties and density moments: {gradient_file}")
    #> load laminar properties and density moments
    laminar_properties = datasets.load_laminar_properties(regress_out_cruvature=('corr-curv' in gradient_file))
    profile_moments = datasets.load_profile_moments()
    laminarprops_and_moments = pd.concat([laminar_properties, profile_moments], axis=1)
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> create spin permtations of hist gradients
    print(f"\tCalculating correlations with spin test")
    coefs, pvals, coefs_null_dist =  helpers.spin_test(
        surface_data_to_spin = laminarprops_and_moments.values, 
        surface_data_target = gradient_maps[:, :n_laminar_gradients])
    #> save null distribution for future reference
    np.savez_compressed(
        gradient_file.replace('gradients_surface.npz', 'correlation_LaminarPropsProfileMoments_null.npz'),
        coefs_null_dist=coefs_null_dist
    )
    #> clean and save the results as txt
    coefs = pd.DataFrame(coefs)
    pvals = pd.DataFrame(pvals)
    coefs.columns = pvals.columns = laminarprops_and_moments.columns
    coefs.index = pvals.index = [f'Laminar-G{gradient_num}' for gradient_num in range(1, n_laminar_gradients+1)]
    association_res_str = f"Association with Hist gradients (spin test)\n--------\nCorrelation coefficients:{coefs.T}\nP-values:{pvals.T}\n"
    print(association_res_str)
    with open(gradient_file.replace('gradients_surface.npz', f'correlation_LaminarPropsProfileMoments.txt'), 'w') as association_res_file:
        association_res_file.write(association_res_str)
    #> regression plots
    #>> parcellate gradients and laminar properties
    parcellated_gradients = helpers.parcellate(gradient_maps, parcellation_name)
    parcellated_laminarprops_and_moments = helpers.parcellate(laminarprops_and_moments.values, parcellation_name)
    #>> add back the columns to parcellated laminar properties
    parcellated_laminarprops_and_moments = pd.DataFrame(
        parcellated_laminarprops_and_moments, 
        )
    parcellated_laminarprops_and_moments.columns = laminarprops_and_moments.columns
    for gradient_num in range(1, n_laminar_gradients+1):
        for ylabel in laminarprops_and_moments.columns[:1]:
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.regplot(
                x=parcellated_gradients.iloc[:,gradient_num-1], 
                y=parcellated_laminarprops_and_moments.loc[:, ylabel],
                color='C3', 
                scatter_kws={'alpha':0.2, 's':5},
                ax=ax)
            sns.despine(offset=10, trim=True, ax=ax)
            ax.set_xlabel(f'G{gradient_num}')
            ax.set_ylabel(ylabel)
            #> add correlation coefficients and p vals on the figure
            text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
            text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
            ax.text(text_x, text_y, 
                    f'r = {coefs.iloc[gradient_num-1][ylabel]:.2f}; $\mathregular{{p_{{spin}}}}$ = {pvals.iloc[gradient_num-1][ylabel]:.2f}',
                    color='C3',
                    size=8,
                    multialignment='left')
            fig.tight_layout()
            fig.savefig(
                gradient_file.replace('gradients_surface.npz', f'correlation_{ylabel.replace(" ","_")}_G{gradient_num}.png'),
                dpi=192
                )

def correlate_disorder_atrophy_maps(gradient_file, n_laminar_gradients, n_perm):
    """
    Calculates the correlation between disorder atrophy maps downloaded from
    ENIGMA toolbox  with laminar gradient `gradient_file` (its first `n_laminar_gradients`). 
    Corrects for spatial autocorrelation by doing spin test with `n_perm` permutations.

    Parameters
    ---------
    gradient_file: (str) input gradient file (.npz format) including an array 'surface' 
                         with the shape n_vert x total_n_gradients
    n_laminar_gradients: (int) number of gradients to associate with cortical types
    n_perm: (int) number of spin permutations
    """
    print(f"Investigating correlation with disorder atrophy maps: {gradient_file}")
    #> load gradient maps
    gradient_maps = np.load(gradient_file)['surface']
    #> load disorder atrophy maps
    parcellated_disorder_atrophy_maps = datasets.load_disorder_atrophy_maps()
    #> project it back to surface
    disorder_atrophy_maps = helpers.deparcellate(parcellated_disorder_atrophy_maps, 'aparc')
    #> create spin permtations of hist gradients
    print(f"\tCalculating correlations with spin test")
    coefs, pvals, coefs_null_dist =  helpers.spin_test(
        surface_data_to_spin = disorder_atrophy_maps, 
        surface_data_target = gradient_maps[:, :n_laminar_gradients], 
        )
    #> save null distribution for future reference
    np.savez_compressed(
        gradient_file.replace('gradients_surface.npz', f'correlation_DisorderAtrophy_null.npz'),
        coefs_null_dist=coefs_null_dist
    )
    #> clean and save the results as txt
    coefs = pd.DataFrame(coefs)
    pvals = pd.DataFrame(pvals)
    coefs.columns = pvals.columns = parcellated_disorder_atrophy_maps.columns
    coefs.index = pvals.index = [f'Laminar-G{gradient_num}' for gradient_num in range(1, n_laminar_gradients+1)]
    association_res_str = f"Association with disorder atrophy maps (spin test)\n--------\nCorrelation coefficients:{coefs.T}\nP-values:{pvals.T}\n"
    print(association_res_str)
    with open(gradient_file.replace('gradients_surface.npz', f'correlation_DisorderAtrophy_null.txt'), 'w') as association_res_file:
        association_res_file.write(association_res_str)
    #> regression plots
    #> parcellate gradients to aparc
    parcellated_gradients = helpers.parcellate(gradient_maps, 'aparc').dropna()
    for gradient_num in range(1, n_laminar_gradients+1):
        for ylabel in parcellated_disorder_atrophy_maps.columns:
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.regplot(
                x=parcellated_gradients.iloc[:,gradient_num-1], 
                y=parcellated_disorder_atrophy_maps.loc[:, ylabel],
                color='C3', 
                scatter_kws={'alpha':0.2, 's':5},
                ax=ax)
            sns.despine(offset=10, trim=True, ax=ax)
            ax.set_xlabel(f'G{gradient_num}')
            ax.set_ylabel(ylabel)
            #> add correlation coefficients and p vals on the figure
            text_x = ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
            text_y = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
            ax.text(text_x, text_y, 
                    f'r = {coefs.iloc[gradient_num-1][ylabel]:.2f}; $\mathregular{{p_{{spin}}}}$ = {pvals.iloc[gradient_num-1][ylabel]:.2f}',
                    color='C3',
                    size=8,
                    multialignment='left')
            fig.tight_layout()
            fig.savefig(
                gradient_file.replace('gradients_surface.npz', f'correlation_{ylabel.replace(" ","_")}_G{gradient_num}.png'),
                dpi=192
                )

def compare_fit_disorder_atrophy_maps(gradient_file):
    """
    Compares fit of gradients 1, 2, 3 and cortical types to
    the atrophy map of individual disorders

    TODO: do spin permutation
    """
    #> load and parcellate the gradients
    gradients = np.load(gradient_file)['surface']
    parcellated_gradients = helpers.parcellate(gradients, 'aparc').dropna()
    #> load cortical types of DK parcels
    parcellated_cortical_types = datasets.load_cortical_types('aparc')
    #> load MPC gradients
    hist_gradients = datasets.load_hist_mpc_gradients()
    parcellated_hist_gradients = helpers.parcellate(hist_gradients, 'aparc')
    #> load and deparcellate the disorders map
    parcellated_disorder_atrophy_maps = datasets.load_disorder_atrophy_maps()
    #> remove adys regions from gradients and disorder maps
    if 'excmask-adys' in gradient_file:
        parcellated_gradients = parcellated_gradients.loc[~parcellated_cortical_types.isin(['ALO','AG', 'DG']), :]
        parcellated_disorder_atrophy_maps = parcellated_disorder_atrophy_maps.loc[~parcellated_cortical_types.isin(['ALO','AG', 'DG']), :]
        n_types = 4
    else:
        n_types = 6
    #> bin gradients to n_types bins, to make it comparable to cortical types
    for g_idx in range(3):
        parcellated_gradients[f'bin{g_idx}'] = pd.cut(parcellated_gradients.loc[:, g_idx], n_types)
    #> calculate adjusted R2 of each disorder atrophy map to Gs or cortical types
    AdjR2s = pd.DataFrame()
    Xs = pd.DataFrame({
        'G1': parcellated_gradients.loc[:, 0],
        'G2': parcellated_gradients.loc[:, 1],
        'G3': parcellated_gradients.loc[:, 2],
        'G1_binned': parcellated_gradients.loc[:, 'bin0'],
        'G2_binned': parcellated_gradients.loc[:, 'bin1'],
        'G3_binned': parcellated_gradients.loc[:, 'bin2'],
        'HistG1': parcellated_hist_gradients.loc[:, 0],
        'HistG2': parcellated_hist_gradients.loc[:, 1],
        'CorticalType': parcellated_cortical_types.cat.codes,
    })
    parcellated_disorder_atrophy_maps = parcellated_disorder_atrophy_maps.loc[Xs.index & parcellated_disorder_atrophy_maps.index]
    Xs = Xs.loc[parcellated_disorder_atrophy_maps.index]
    df = pd.concat([parcellated_disorder_atrophy_maps, Xs], axis=1)
    X_names_dict = {
        'G1': ['G1'],
        'G2': ['G2'],
        'G3': ['G3'],
        'G1-3': ['G1', 'G2', 'G3'],
        'G1 binned': ['C(G1_binned)'],
        'G2 binned': ['C(G2_binned)'],
        'G3 binned': ['C(G3_binned)'],
        'Hist G1': ['HistG1'],
        'Hist G2': ['HistG2'],
        'Hist G1-2': ['HistG1', 'HistG2'],
        'Cortical types': ['CorticalType'],
        'Cortical types (cat)': ['C(CorticalType)']
    }
    include_in_plot = ['G1', 'Cortical types', 'Cortical types (cat)']
    res_str = ""
    for disorder in parcellated_disorder_atrophy_maps.columns:
        for list_name , X_names_list in X_names_dict.items():
            formula = f'{disorder} ~ {" + ".join(X_names_list)}'
            res = smf.ols(formula=formula, data=df).fit()
            res_str += f"\n-----------\n{formula}\n{res.summary()}\n"
            AdjR2s.loc[list_name, disorder] = res.rsquared_adj
    res_str = f"Adj R2s:\n{AdjR2s}\n\nModels details:\n{res_str}"
    with open(gradient_file.replace('gradients_surface.npz', 'compare_disorder_fit.txt'), 'w') as outfile:
        outfile.write(res_str)
    #> plot
    AdjR2s_toplot = AdjR2s.loc[include_in_plot, :]
    for disorder in AdjR2s_toplot.columns:
        fig, ax = plt.subplots(1, figsize=(3, 1.5))
        sns.barplot(
            data=AdjR2s_toplot,
            x=disorder,
            y=AdjR2s_toplot.index,
            color='C0',
            alpha=0.25,
            ax=ax
        )
        ax.set_xlim((AdjR2s.values.min(), AdjR2s.values.max()))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([]);
        ax.set_xlabel('');
        fig.tight_layout()
        fig.savefig(
            gradient_file.replace('gradients_surface.npz', f'compare_disorder_fit_{disorder}'),
            dpi=192
        )
    

#> run all functions
# gradient_files = glob.glob(os.path.join(DATA_DIR, 'result', '*parcor*', 'gradients_surface.npz'))
gradient_files = [os.path.join(DATA_DIR, 'result', 'input-thickness_parc-sjh_approach-dm_metric-parcor_parcel_excmask-adys', 'gradients_surface.npz')]
for gradient_file in gradient_files:
    print("Gradient:", gradient_file)
    # associate_cortical_types(gradient_file)
    # associate_yeo_networks(gradient_file)
    correlate_hist_gradients(gradient_file, n_laminar_gradients=3, n_perm=1000)
    correlate_laminar_properties_and_moments(gradient_file, n_laminar_gradients=3, n_perm=1000)
    # correlate_disorder_atrophy_maps(gradient_file, n_laminar_gradients=3, n_perm=1000)
    # compare_fit_disorder_atrophy_maps(gradient_file)