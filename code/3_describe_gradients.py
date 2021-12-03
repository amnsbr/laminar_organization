"""
Descriptive analyses on the gradients, including:
1. Plotting the gradient surfaces
2. Plotting binned laminar profiles

TODO: consider combining these with 2_create_gradients
"""
import helpers
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#> specify the data dir and create gradients and matrices subfolders
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')

def plot_gradients(gradient_file, n_gradients=3):
    """
    Plots the first `n_gradients` of `gradient_file`
    """
    print("\tPlotting on surface")
    #> loading gradient map
    gradient_maps = np.load(gradient_file)['surface']
    for gradient_num in range(1, n_gradients+1):
        helpers.plot_on_bigbrain_nl(
            gradient_maps[:, gradient_num-1],
            filename=gradient_file.replace('.npz', f'_g{gradient_num}.png')
        )

def plot_binned_laminar_profile(gradient_file, n_gradients=3):
    """
    Plots the relative laminar thickness (TODO: and density) of 10 bins of
    the first `n_gradients` of `gradient_file`
    """
    print("\tPlotting binned laminar profiles")
    #> determine input type
    re.match(r".*_input-([a-z|-]+)_*", gradient_file).groups()[0]
    regress_out_curvature = 'corr-curv' in gradient_file
    #> loading gradient map
    gradient_maps = np.load(gradient_file)['surface']
    #> loading thickness and density data and parcellating them
    laminar_thickness = helpers.read_laminar_thickness(regress_out_curvature=regress_out_curvature)
    # laminar_density = helpers.read_laminar_density()
    #> parcellate the data
    parcellated_gradients = helpers.parcellate(gradient_maps, 'sjh')
    parcellated_laminar_thickness = helpers.parcellate(laminar_thickness, 'sjh')
    parcellated_laminar_thickness = helpers.concat_hemispheres(parcellated_laminar_thickness, dropna=False)
    # re-normalize small deviations from sum=1 because of parcellation
    parcellated_laminar_thickness /= parcellated_laminar_thickness.sum(axis=1)
    for gradient_num in range(1, n_gradients+1):
        binned_parcels_laminar_thickness = parcellated_laminar_thickness.copy()
        binned_parcels_laminar_thickness['bin'] = pd.cut(parcellated_gradients[gradient_num-1], 10)
        #> calculate average laminar thickness at each bin
        bins_laminar_thickness = binned_parcels_laminar_thickness.groupby('bin').mean().reset_index(drop=True)
        bins_laminar_thickness.columns = [f'Layer {idx+1}' for idx in range(6)]
        #> reverse the columns so that in the plot Layer 6 is at the bottom
        bins_laminar_thickness = bins_laminar_thickness[bins_laminar_thickness.columns[::-1]]
        #> normalize to sum of 1 at each bin
        bins_laminar_thickness = bins_laminar_thickness.divide(bins_laminar_thickness.sum(axis=1), axis=0)
        #> plot the relative thickness of layers 6 to 1
        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = 'Blues'
        ax.bar(
            x = bins_laminar_thickness.index,
            height = bins_laminar_thickness['Layer 6'],
            width = 0.95,
            color=plt.cm.get_cmap(cmap)(6/6),
            )
        for layer_num in range(5, 0, -1):
            ax.bar(
                x = bins_laminar_thickness.index,
                height = bins_laminar_thickness[f'Layer {layer_num}'],
                width = 0.95,
                bottom = bins_laminar_thickness.cumsum(axis=1)[f'Layer {layer_num+1}'],
                color=plt.cm.get_cmap(cmap)(layer_num/6),
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'G{gradient_num} bins')
        ax.set_ylabel('Relative laminar thickness')
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        fig.tight_layout()
        fig.savefig(gradient_file.replace('gradients_surface.npz', f'binned_profile_g{gradient_num}.png'), dpi=192)
        clfig = helpers.make_colorbar(
            parcellated_gradients[gradient_num-1].min(), 
            parcellated_gradients[gradient_num-1].max(), 
            orientation='horizontal', figsize=(6,4))
        clfig.savefig(gradient_file.replace('gradients_surface.npz', f'binned_profile_g{gradient_num}_clbar.png'), dpi=192)

#> describe all gradients
gradient_files = glob.glob(os.path.join(DATA_DIR, 'gradient', '*', 'gradients_surface.npz'))
for gradient_file in gradient_files:
    print("Gradient file:", gradient_file)
    plot_gradients(gradient_file)
    plot_binned_laminar_profile(gradient_file)
