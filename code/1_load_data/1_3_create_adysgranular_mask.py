"""
Create masks of bigbrain space including agranular and dysgranular region.
Since the data is processed parcellated, these masks also take the parcellation
into account and include parcels that have > e.g. 10% overlap with the
a-/dysgranular regions.
"""
import os
import pandas as pd
import numpy as np
import nilearn.surface

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', '..', 'data')

# Config
TOLERABLE_ADYS_IN_PARCELS = 0.1 # how much adysgranular vertices are allowed to be in parcels that are not masked out

#> load the economo parcellation
lh_economo = nilearn.surface.load_surf_data(
    os.path.join(DATA_DIR, 'parcellations', 'tpl-bigbrain_hemi-L_desc-economo_parcellation.label.gii')
    )
rh_economo = nilearn.surface.load_surf_data(
    os.path.join(DATA_DIR, 'parcellations', 'tpl-bigbrain_hemi-R_desc-economo_parcellation.label.gii')
    )

#> get the indices of parcels that are agranular, dysgranular, allocortex or NaN (corpus callosum and unknown)
cortical_types = pd.read_csv(
    os.path.join(DATA_DIR, 'parcellated_surface', 'economo_cortical_types.csv')
    )
adysgranular_regions = (cortical_types[
    cortical_types['CorticalType']
    .isin(['AG', 'DG', 'ALO', np.NaN])
    ].index.tolist())

print("These regions belong to the agranular or dysgranular cortical type or the allocortex:")
print(cortical_types.loc[adysgranular_regions, 'Label'].tolist())

#> create a mask of a-/dysgranular vertices
lh_adysgranular_mask = np.in1d(lh_economo, adysgranular_regions)
rh_adysgranular_mask = np.in1d(rh_economo, adysgranular_regions)

#> load parcellation maps
lh_sjh_parcellation = nilearn.surface.load_surf_data(
    os.path.join(DATA_DIR, 'parcellations', 'tpl-bigbrain_hemi-L_desc-sjh_parcellation.label.gii'))
rh_sjh_parcellation = nilearn.surface.load_surf_data(
    os.path.join(DATA_DIR, 'parcellations', 'tpl-bigbrain_hemi-R_desc-sjh_parcellation.label.gii'))

#> calculate the proportion of adysgranular vertices in each parcel
lh_parcels_adys_proportion = pd.DataFrame(
    {'parcel': lh_sjh_parcellation, 'adys': lh_adysgranular_mask}
    ).groupby('parcel')['adys'].mean()
rh_parcels_adys_proportion = pd.DataFrame(
    {'parcel': rh_sjh_parcellation, 'adys': rh_adysgranular_mask}
    ).groupby('parcel')['adys'].mean()

#> determine the parcels that pass the threshold and need to be masked
lh_adys_parcels = lh_parcels_adys_proportion[
    lh_parcels_adys_proportion > TOLERABLE_ADYS_IN_PARCELS
    ].index.to_numpy()
rh_adys_parcels = rh_parcels_adys_proportion[
    rh_parcels_adys_proportion > TOLERABLE_ADYS_IN_PARCELS
    ].index.to_numpy()

#> create an extended mask of adysgranular regions
lh_adysgranular_sjh_ext_mask = np.in1d(lh_sjh_parcellation, lh_adys_parcels)
rh_adysgranular_sjh_ext_mask = np.in1d(rh_sjh_parcellation, rh_adys_parcels)

#> save the mask
np.save(
    os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-adysgranular_sjh_ext_mask.npy'),
    lh_adysgranular_sjh_ext_mask
)
np.save(
    os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-adysgranular_sjh_ext_mask.npy'),
    rh_adysgranular_sjh_ext_mask
)
print("Masked saved in data/surface/tpl-bigbrain_hemi-*_desc-adysgranular_sjh_ext_mask.npy")