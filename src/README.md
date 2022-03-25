The `src` directory includes local and external input data used in the analyses.

|File/Directory Name|Description|Source|Comments|
|---|---|---|---|
|`*h_aparc.annot`|DK parcellation in fsaverage space|[DevelopmentalImagingMCRI / freesurfer_statsurf_display](https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/tree/master/fsaverage_fs6/label)| |
|`*h_economo.annot`|von Economo parcellation in fsaverage space|[DevelopmentalImagingMCRI / freesurfer_statsurf_display](https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/tree/master/fsaverage_fs6/label)| |
|`economo_cortical_types.csv`|Cortical type of each von Economo region|Local|Created manually based on [García-Cabezas 2020](https://doi.org/10.3389/fnana.2020.576015). Labels correspond to `*h_economo.annot`|
|`*h_schaefer*.annot`|Schaefer-N parcellation in fsaverage space|[ThomasYeoLab / CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label)|Originally named `*h.Schaefer2018_*Parcels_7Networks_order.annot`|
|`*h_sjh.annot`|SJH parcellation in fsaverage space|[MICA-MNI / micaopen](https://github.com/MICA-MNI/micaopen/tree/master/MPC/maps)| |
|`tpl-bigbrain_hemi-*_desc-*_parcellation.label.gii`|Parcellations in bigbrain surface space|Local|Created using `code/local/transform_to_bigbrain.sh`|
|`tpl-bigbrain_hemi-*_desc-*_parcellation_centers.csv`|Center indices of each parcel in bigbrain surface|Local|Created using `code/helpers.py`:`get_parcel_center_indices`|
|`tpl-MNI152_desc-schaefer*_parcellation.nii.gz`|Schaefer-N parcellation in MNI space|[ThomasYeoLab / CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI)|Original name: `Schaefer2018_*Parcels_7Networks_order_FSLMNI152_1mm.nii.gz`|
|`tpl-MNI152_desc-sjh_parcellation.nii.gz`|SJH parcellation in MNI space|Local|Created using `code/local/sjh_to_mni.m`|
|`tpl-MNI152_desc-aparc_parcellation.nii.gz`|DK parcellation in MNI space|Local|Created based on the DK parcellation in abagen after removing subcortex|
|`tpl-bigbrain_desc-profiles.txt`|Total-depth BigBrain density profiles (50 samples)|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_hemi_*_desc-Hist_G*.txt`|Histology gradients of the BigBrain|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)|Based on [Paquola 2019](https://doi.org/10.1371/journal.pbio.3000284)|
|`tpl-bigbrain_hemi_*_desc-layer*_thickness.txt`|Thickness of individual layers in BigBrain|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)|Based on [Wagstyl 2020](https://doi.org/10.1371/journal.pbio.3000678)|
|`tpl-bigbrain_hemi_*_desc-layer*_profiles_nsurf-10.npz`|Layer-specific BigBrain density profiles (10 samples / layer)|Local|Created using `code/local/create_laminar_density_profiles.sh`|
|`tpl-bigbrain_hemi_*_desc-{white/mid/pial}.surf.gii`|BigBrain cortical surface mesh|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_hemi_*_desc-mid.surf.inflate.gii`|Inflated BigBrain mid-cortical surface mesh|Local|Created using FreeSurfer in `code/local/inflate_bigbrain.sh`|
|`tpl-bigbrain_hemi_*_desc-sphere_rot_fsaverage.surf.gii`|BigBrain surface sphere used for spin permutation|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_hemi_*_desc-{pial/white}.area.npy`|Pial/WM surface area of BigBrain|Local|Created using CIVET in `code/local/calculate_surface_area.py`|
|`tpl-bigbrain_hemi-*_desc-Yeo2011_7Networks_N1000.label.gii`|Yeo 7 functional networks in BigBrain surface space|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`spin_batches/`|BigBrain surface sphere spun for permutation testing|Local|Created in `code/helpers.py` > `create_bigbrain_spin_permutations`. Each batch includes 20 random spins.|
|`PET_nifti_images`|Receptors volumetric PET maps|[netneurolab / hansen_receptors](https://github.com/netneurolab/hansen_receptors/tree/main/data/PET_nifti_images)| |
|`PET_nifti_images_metadata.csv`|Metdata for volumetric PET maps|Local|Created based on filenames and the information provided in Hansen et al. 2021 and the source papers for some|
|`ahba_parc-*_frozen-*`|Expression of genes in each parcellation based on AHBA|Local|Created using `code/local/fetch_ahba.py` via [abagen](https://abagen.readthedocs.io/en/stable/)|
|`celltypes_PSP.csv`|List of genes associated with each cell type|[jms290 / PolySyn_MSNs](https://raw.githubusercontent.com/jms290/PolySyn_MSNs/master/Data/AHBA/celltypes_PSP.csv)|Based on [Seidlitz 2020](https://www.nature.com/articles/s41467-020-17051-5)|