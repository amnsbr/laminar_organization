The `src` directory includes external input data used in the analyses, as well as sources created by the functions in `code` that are saved here due to the high computational cost of creating them (such as spin permutations) or external dependencies that are difficult to reproduce (output of the scripts in `code/local`).

|File/Directory Name|Description|Source|Comments|
|---|---|---|---|
|`ahba_*`|Expression of genes in each parcellation based on AHBA|Local|Created using `helpers.fetch_ahba_data`|
|`economo_cortical_types.csv`|Cortical type of each von Economo region|Local|Created manually based on [García-Cabezas 2020](https://doi.org/10.3389/fnana.2020.576015) (Tables 4-7). Labels correspond to `*h_economo.annot`|
|`{exc/inh}_subtypes_genes_Lake2016.csv`|List of genes associated with excitatory and inhibitory neuronal subtypes and their weights|Local|Based on Table S5 of [Lake 2016](https://doi.org/10.1126/science.aaf1204)|
|`{hcp/mics}_rDCM_sch400.mat`|Effective connectivity matrices|[caseypaquola / DMN](https://github.com/caseypaquola/DMN/tree/main/data)|From [Paquola 2021](https://www.biorxiv.org/content/10.1101/2021.11.22.469533v1)|
|`{L/R}.human-to-macaque.sphere.reg.32k_fs_LR.surf.gii`|Human to macaque registration|[TingsterX/alignment_macaque-human](https://github.com/TingsterX/alignment_macaque-human)| |
|`{L/R}.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii`|Human to macaque registration|[TingsterX/alignment_macaque-human](https://github.com/TingsterX/alignment_macaque-human)| |
|`*h_aparc.*`|DK parcellation in fsa5/fsa7 space (annot/gifti)|[DevelopmentalImagingMCRI / freesurfer_statsurf_display](https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/tree/master/fsaverage_fs6/label)|Annot <-> Gifti transformation done in `helpers.py` for all parcellations|
|`*h_brodmann.*`|Brodmann regions in fsa5/fsa7 space (annot/gifti)|[Pijnenburg 2021](https://www.sciencedirect.com/science/article/pii/S1053811921005504#sec0031)|Supplementary zip file|
|`*h_economo.*`|von Economo parcellation in fsa5/fsa7 space (annot/gifti)|[DevelopmentalImagingMCRI / freesurfer_statsurf_display](https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/tree/master/fsaverage_fs6/label)| |
|`*h_schaefer*.*`|Schaefer-N parcellation in fsa5/fsa7 space (annot/gifti)|[ThomasYeoLab / CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label)|Originally named `*h.Schaefer2018_*Parcels_7Networks_order.annot`|
|`*h_yan*.*`|Homotopic version of Schaefer-N parcellation in fsa5/fsa7 space (annot/gifti)|[ThomasYeoLab / CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/FreeSurfer/fsaverage/label/yeo7)|Originally named `*h.*Parcels_Yeo2011_7Networks.annot`|
|`*h_sjh.*`|SJH parcellation in fsa5/fsa7 space (annot/gifti)|[MICA-MNI / micaopen](https://github.com/MICA-MNI/micaopen/tree/master/MPC/maps)| |
|`macaque_hierarchy.pscalar.nii`|Laminar-based hierarchy of macaque (M132 parcellation)|[Burt 2018](https://doi.org/10.1038/s41593-018-0195-0)|[Associated data in BALSA](https://balsa.wustl.edu/study/Kx5n)|
|`MacaqueYerkes19.*`|Surface meshes of macaque in fs_LR 32k space|[TingsterX/alignment_macaque-human](https://github.com/TingsterX/alignment_macaque-human)| |
|`S1200.{L/R}.{pial_MSMAll/sphere}.32k_fs_LR.surf.gii`|Surface meshes of human in fs_LR 32k space|[TingsterX/alignment_macaque-human](https://github.com/TingsterX/alignment_macaque-human)| |
|`source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-*_feature.func.gii`|HCP S1200 group-averaged myelin map in fsLR space|[netneurolab / neuromaps](https://netneurolab.github.io/neuromaps)| |
|`scov_Valk2020.mat`|HCP S1200 structural covariance matrix in schaefer400|[Valk 2020](https://doi.org/10.1126/sciadv.abb3417)|Requested from the author|
|`tpl-bigbrain_desc-laminar_thickness_*.npz`|Thickness of individual layers in BigBrain after smoothing|Local|Created using `datasets.load_laminar_thickness`|
|`tpl-bigbrain_desc-profiles.txt`|Total-depth BigBrain density profiles (50 samples)|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_desc-spin_indices_downsampled_n-1000.npz`|1000 spin permutations of vertices on bigbrain downsampled surface|Local|Created using `helpers.create_bigbrain_spin_permutations`|
|`tpl-bigbrain_*_desc-rotated_parcels_n-1000.txt`|1000 spin permutations of a given parcellation|Local|Created using `helpers.get_rotated_parcels`|
|`tpl-bigbrain_hemi-*_desc-*_parcellation.label.gii`|Parcellations in bigbrain surface space|Local|Created using `code/local/transform_to_bigbrain.sh`|
|`tpl-bigbrain_hemi_*_desc-Hist_G*.txt`|Histology gradients of the BigBrain|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)|Based on [Paquola 2019](https://doi.org/10.1371/journal.pbio.3000284)|
|`tpl-bigbrain_hemi_*_desc-layer*_thickness.txt`|Thickness of individual layers in BigBrain|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)|Based on [Wagstyl 2020](https://doi.org/10.1371/journal.pbio.3000678)|
|`tpl-bigbrain_hemi_*_desc-layer*_profiles_nsurf-10.npz`|Layer-specific BigBrain density profiles (10 samples / layer)|Local|Created using `code/local/create_laminar_density_profiles.sh`|
|`tpl-bigbrain_hemi_*_desc-{white/mid/pial}.surf.gii`|BigBrain cortical surface mesh|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_hemi_*_desc-mid.surf.inflate.gii`|Inflated BigBrain mid-cortical surface mesh|Local|Created using FreeSurfer in `code/local/inflate_bigbrain.sh`|
|`tpl-bigbrain_hemi_*_desc-pial_downsampled_{inflated/orig/sphere}.surf.gii`|Bigbrain downsampled pial mesh in different versions|Local|Created using `datasets.load_mesh_paths`|
|`tpl-bigbrain_hemi_*_desc-sphere_rot_fsaverage.surf.gii`|BigBrain surface sphere used for spin permutation|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_hemi_*_desc-{pial/white}.area.npy`|Pial/WM surface area of BigBrain|Local|Created using CIVET in `code/local/calculate_surface_area.py`|
|`tpl-bigbrain_hemi-*_desc-Yeo2011_7Networks_N1000.label.gii`|Yeo 7 functional networks in bigbrain surface space|[caseypaquola / BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp)| |
|`tpl-bigbrain_hemi-*_desc-hcp1200_myelinmap.shape.gii`|HCP S1200 group-averaged myelin map in bigbrain space|[netneurolab / neuromaps](https://netneurolab.github.io/neuromaps)|Transformed to BigBrain space using `code/local/transform_to_bigbrain.sh`|
|`tpl-*_hemi-*{_downsampled}_parc-*_desc-centers.csv`|Center indices of each parcel for different parcellations and cortical surfaces|Local|Created using `helpers.get_parcel_center_indices`|
|`tpl-bigbrain_parc-*_desc-cellular_*.csv`|Laminar cellular features of selected samples in BigBrain|[Ebrains](https://search.kg.ebrains.eu/instances/f06a2fd1-a9ca-42a3-b754-adaa025adb10) & [FZJ-INM1-BDA /siibra-python](https://siibra-python.readthedocs.io/en/latest/index.html#)|Loaded using `datasets.fetch_laminar_cellular_features`|
|`tpl-MNI152_desc-schaefer*_parcellation.nii.gz`|Schaefer-N parcellation in MNI space|[ThomasYeoLab / CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI)|Original name: `Schaefer2018_*Parcels_7Networks_order_FSLMNI152_1mm.nii.gz`|
|`tpl-MNI152_desc-sjh_parcellation.nii.gz`|SJH parcellation in MNI space|Local|Created using `code/local/sjh_to_mni.m`|
|`tpl-MNI152_desc-aparc_parcellation.nii.gz`|DK parcellation in MNI space|Local|Created based on the DK parcellation in abagen after removing subcortex|
|`Yerkes19_Parcellations_v2.32k_fs_LR.dlabel.nii`|Macaque parcellations (including M132)|[Donahue 2016](https://doi.org/10.1523/JNEUROSCI.0493-16.2016)|[Associated data in BALSA](https://balsa.wustl.edu/reference/976nz)