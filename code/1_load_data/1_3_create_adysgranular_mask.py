import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import nilearn.surface

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', '..', 'data')

def create_adysgranular_mask(parcellation_name=None, tolerable_adys_in_parcels=0.1):
    """
    Create masks of bigbrain space including agranular and dysgranular region.
    When the data is processed parcellated, these masks also take the parcellation
    into account and include parcels that have e.g. > 10% overlap with the
    a-/dysgranular regions

    Parameters
    ---------
    parcellation_name: (None or str) the parcellation should be saved in parcellations folder
                        with the name 'tpl-bigbrain_hemi-L_desc-{parcellation_name}_parcellation.label.gii'
    tolerable_adys_in_parcels: (float) % overlap of parcels with a-/dysgranular allowed
    """
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
    for hem in ['L', 'R']:
        #> load the economo parcellation
        economo_map = nilearn.surface.load_surf_data(
            os.path.join(DATA_DIR, 'parcellations', f'tpl-bigbrain_hemi-{hem}_desc-economo_parcellation.label.gii')
            )        
        #> create a mask of a-/dysgranular vertices
        adysgranular_mask = np.in1d(economo_map, adysgranular_regions)

        if parcellation_name:
            #> load parcellation maps
            parcellation_map = nilearn.surface.load_surf_data(
                os.path.join(DATA_DIR, 'parcellations', f'tpl-bigbrain_hemi-{hem}_desc-{parcellation_name}_parcellation.label.gii')
                )
            #> calculate the proportion of adysgranular vertices in each parcel
            parcels_adys_proportion = pd.DataFrame(
                {'parcel': parcellation_map, 'adys': adysgranular_mask}
                ).groupby('parcel')['adys'].mean()

            #> determine the parcels that pass the threshold and need to be masked
            adys_parcels = (parcels_adys_proportion[
                parcels_adys_proportion > tolerable_adys_in_parcels
                ].index.to_numpy())

            #> create an extended mask of adysgranular regions
            adysgranular_mask = np.in1d(parcellation_map, adys_parcels)

        #> save the mask
        mask_filepath = os.path.join(
            DATA_DIR, 'surface', 
            f'tpl-bigbrain_hemi-{hem}_desc-adysgranular_mask_parcellation-{str(parcellation_name).lower()}_thresh_{tolerable_adys_in_parcels}.npy'
            )
        np.save(
            mask_filepath,
            adysgranular_mask
        )
        print(f"Masked saved in {mask_filepath}")


parser = ArgumentParser()
parser.add_argument("-p", "--parcellation", dest="parcellation",
                    help="Parcellation scheme to use", default=None)
parser.add_argument("-t", "--thresh", dest="threshold", default=0.1,
                    help="%\ overlap of parcels with a-/dysgranular allowed")

args = parser.parse_args()
create_adysgranular_mask(args.parcellation, args.threshold)