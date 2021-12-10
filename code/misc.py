import helpers
import numpy as np
import matplotlib.pyplot as plt


def plot_parcels_laminar_profile(parcellation_name, palette='bigbrain'):
    """
    Plots the laminar profile of all parcels in a stacked bar plot

    Parameters
    ---------
    parcellation_name: (str)
    palette: (str)
        - bigbrain: layer colors on BigBrain web viewer
        - wagstyl: layer colors on Wagstyl 2020 paper
    """
    #> load and parcellate laminar data
    laminar_data = helpers.read_laminar_thickness(
                    exc_masks=None, 
                    normalize_by_total_thickness=True, 
                    regress_out_curvature=False
                )
    concat_laminar_data = np.concatenate([laminar_data['L'], laminar_data['R']], axis=0)
    parcellated_concate_laminar_data = helpers.parcellate(concat_laminar_data, parcellation_name)
    #> plot the relative thickness of layers 6 to 1
    fig, ax = plt.subplots(figsize=(100, 20))
    if palette == 'bigbrain':
        colors = ['#abab6b', '#dabcbc', '#dfcbba', '#e1dec5', '#66a6a6','#d6c2e3'] # layer 1 to 6
    elif palette == 'wagstyl':
        colors = ['#3a6aa6ff', '#f8f198ff', '#f9bf87ff', '#beaed3ff', '#7fc47cff','#e31879ff'] # layer 1 to 6
    ax.bar(
        x = parcellated_concate_laminar_data.index,
        height = parcellated_concate_laminar_data.iloc[:, 5],
        width = 2,
        color=colors[5]
        )
    for col_idx in range(4, -1, -1):
        ax.bar(
            x = parcellated_concate_laminar_data.index,
            height = parcellated_concate_laminar_data.iloc[:, col_idx],
            width = 2,
            bottom = parcellated_concate_laminar_data.iloc[:, ::-1].cumsum(axis=1).loc[:, col_idx+1],
            color=colors[col_idx]
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for _, spine in ax.spines.items():
        spine.set_visible(False)