% Projects SJH parcellation in fsaverage surface space to
% MNI152 volumetric space

% Setup:
% 1. Make sure these are included in the Matlab path before running:
% * CBIG repository https://github.com/ThomasYeoLab/CBIG/
%   - /stable_projects/registration/Wu2017_RegistrationFusion
%   - /utilities/matlab/surf
%   - /external_packages/SD/SDv1.5.1-svn593
% * Freesurfer matlab functions
% 2. $FREESURFER_HOME must be in env variables
% 3. On mac, the precompiled mex files existing in CBIG do not work
%  and need to be compiled locally. For this:
% a. Setup mex (which requires Xcode on mac):
% % > mex -setup
% b. Run mex on these files in $CBIG/external_packages/SD:
%  [kdtree.c, kdtreeidx.c, MARS_computeMeshFaceAreas.c,
%   MARS_computeVertexDistSq2Nbors.c, MARS_convertFaces2FacesOfVert.c,
%   MARS_convertFaces2VertNbors.c]
%   * for the last two, it may be required to move the function
%     `index_2D_array` to top of the .c file!
% 4. Input surface .annot files must be in the same directory as
%  this matlab script.

% Run the transformation:

% Specify the path to mask and mask_input
% Note: These are the defaults used by CBIG_RF_projectfsaverage2Vol_single
% but on mac with FS 7.2 I have to input these paths manually
% since if I keep this empty, the script will look for these
% in final_warps_FS (and not final_warps_FS5.3) and similarly
% for liberal_cortex_masks, which does not exist
CBIG_CODE_DIR = '/Users/asaberi/Desktop/CBIG_selected/';
map = strcat(CBIG_CODE_DIR, ...,
    'stable_projects/registration/Wu2017_RegistrationFusion/bin/final_warps_FS5.3/allSub_fsaverage_to_FSL_MNI152_FS4.5.0_RF_ANTs_avgMapping.prop.mat');
mask_input = strcat(CBIG_CODE_DIR, ...,
    'stable_projects/registration/Wu2017_RegistrationFusion/bin/liberal_cortex_masks_FS5.3/FSL_MNI152_FS4.5.0_cortex_estimate.nii.gz');

% Load .annot files in fsaverage space
% see https://surfer.nmr.mgh.harvard.edu/fswiki/AnnotFiles
% on what each variable means
[lh_v, lh_input, lh_ct] = read_annotation('lh_sjh.annot');
[rh_v, rh_input, rh_ct] = read_annotation('rh_sjh.annot');

% Convert original unique ids in parcellation file
% which are unique numbers resulted from an arithmetic
% operation on the color codes of each parcel,
% to SJH parcel ids ('sjh_{id}') to make it compatible
% with the way its surface parcel is loaded
%> remove 'sjh_' from struct_names and convert it to integer
lh_sjh_ids = str2double(erase(lh_ct.struct_names, 'sjh_'));
rh_sjh_ids = str2double(erase(rh_ct.struct_names, 'sjh_'));
%> create a map object (similar to dictionary in Python)
% for mapping original ids in *h_input (located in the 
% last column of colortable) to sjh ids
lh_map = containers.Map(lh_ct.table(:, 5), lh_sjh_ids);
rh_map = containers.Map(rh_ct.table(:, 5), rh_sjh_ids);
%> use the map object on the parcellation maps (*h_input)
lh_input = cell2mat(values(lh_map, num2cell(lh_input)));
rh_input = cell2mat(values(rh_map, num2cell(rh_input)));
%> reshape the maps to 1xN arrays
% (CBIG_RF_projectfsaverage2Vol_single requirement)
lh_input = transpose(lh_input);
rh_input = transpose(rh_input);

% Project fsaverage to MNI
[projected, projected_seg] = CBIG_RF_projectfsaverage2Vol_single(lh_input, rh_input, 'nearest', map, mask_input);
% Write projected to .nii.gz files
% note that we don't need projected_seg because the parcel ids in right
% hemishpere are already unique and there's no need to add 1000 to them
% to make them distinct from the left hemisphere.
MRIwrite(projected, 'tpl-MNI152_desc-sjh_parcellation.nii.gz');