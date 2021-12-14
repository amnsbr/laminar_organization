import helpers
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_parcels_laminar_profile(parcellation_name, exc_masks=None, palette='bigbrain'):
    """
    Plots the laminar profile of all parcels in a stacked bar plot

    Parameters
    ---------
    parcellation_name: (str)
    exc_masks: (str) path to the exclusion mask
    palette: (str)
        - bigbrain: layer colors on BigBrain web viewer
        - wagstyl: layer colors on Wagstyl 2020 paper
    """
    #> load and parcellate laminar data
    laminar_data = helpers.read_laminar_thickness(
                    exc_masks=exc_masks, 
                    normalize_by_total_thickness=True, 
                    regress_out_curvature=False
                )
    concat_laminar_data = np.concatenate([laminar_data['L'], laminar_data['R']], axis=0)
    parcellated_concat_laminar_data = helpers.parcellate(concat_laminar_data, parcellation_name)
    #> remove NaNs and reindex and renormalize
    parcellated_concat_laminar_data = parcellated_concat_laminar_data.dropna().reset_index(drop=True)
    parcellated_concat_laminar_data = parcellated_concat_laminar_data.divide(
        parcellated_concat_laminar_data.sum(axis=1),
        axis=0
        )
    #> plot the relative thickness of layers 6 to 1
    fig, ax = plt.subplots(figsize=(100, 20))
    if palette == 'bigbrain':
        colors = ['#abab6b', '#dabcbc', '#dfcbba', '#e1dec5', '#66a6a6','#d6c2e3'] # layer 1 to 6
    elif palette == 'wagstyl':
        colors = ['#3a6aa6ff', '#f8f198ff', '#f9bf87ff', '#beaed3ff', '#7fc47cff','#e31879ff'] # layer 1 to 6
    ax.bar(
        x = parcellated_concat_laminar_data.index,
        height = parcellated_concat_laminar_data.iloc[:, 5],
        width = 1,
        color=colors[5]
        )
    for col_idx in range(4, -1, -1):
        ax.bar(
            x = parcellated_concat_laminar_data.index,
            height = parcellated_concat_laminar_data.iloc[:, col_idx],
            width = 1,
            bottom = parcellated_concat_laminar_data.iloc[:, ::-1].cumsum(axis=1).loc[:, col_idx+1],
            color=colors[col_idx]
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            helpers.DATA_DIR, 'parcellated_surface',
            f'brain_laminar_profile_parc-{parcellation_name}{"_excmask" if exc_masks else ""}'
            ))

for parcellation_name in ['sjh', 'schaefer400']:
    adysgranular_masks = {
        'L': os.path.join(
            helpers.DATA_DIR, 'surface',
            f'tpl-bigbrain_hemi-L_desc-adysgranular_mask_parcellation-{parcellation_name}_thresh_0.1.npy'
        ),
        'R': os.path.join(
            helpers.DATA_DIR, 'surface',
            f'tpl-bigbrain_hemi-R_desc-adysgranular_mask_parcellation-{parcellation_name}_thresh_0.1.npy'
        )
    }
    for exc_masks in [None, adysgranular_masks]:
        plot_parcels_laminar_profile(parcellation_name, exc_masks=exc_masks)