The `src` directory includes local and external input data used in the analyses.

|Filename|Description|Source|Comments|
|---|---|---|---|
|`*h_aparc.annot`|DK parcellation in fsaverage space|[DevelopmentalImagingMCRI / freesurfer_statsurf_display](https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/tree/master/fsaverage_fs6/label)| |
|`*h_economo.annot`|von Economo parcellation in fsaverage space|[DevelopmentalImagingMCRI / freesurfer_statsurf_display](https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/tree/master/fsaverage_fs6/label)| |
|`economo_cortical_types.csv`|Cortical type of each von Economo region|Local|Created manually based on [García-Cabezas 2020](https://doi.org/10.3389/fnana.2020.576015). Labels correspond to `*h_economo.annot`|
|`*h_schaefer400.annot`|Schaefer-400 parcellation in fsaverage space|[ThomasYeoLab / CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label)|Originally named `*h.Schaefer2018_400Parcels_7Networks_order.annot`|
|`*h_sjh.annot`|SJH-1200 parcellation in fsaverage space|[MICA-MNI / micaopen](https://github.com/MICA-MNI/micaopen/tree/master/MPC/maps)| |
|`tpl-bigbrain_hemi-*_desc-*_parcellation.label.gii`|Parcellations in bigbrain surface space|Local|Created using `code/local/transform_to_bigbrain.sh`|
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