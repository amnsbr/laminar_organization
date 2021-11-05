"""
Downloads parcellations, spaces and laminar thickness data into 'src' folder
and copies those that need no preprocessing to the 'data' folder
"""
import os
import shutil
import helpers

#> Create and change path to 'src' folder
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
SRC_DIR = os.path.abspath(os.path.join(cwd, '..', 'src'))
os.chdir(SRC_DIR)
os.makedirs(SRC_DIR, exist_ok=True)
#> Create 'data' and its subfolders
DATA_DIR = os.path.abspath(os.path.join(cwd, '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)
for subfolder in ['parcellation', 'parcellated_surface', 'surface', 'gradient', 'matrix']:
	os.makedirs(os.path.join(DATA_DIR, subfolder), exist_ok=True)

for hem in ['L', 'R']:
	#> Bigbrain space (copy in data/)
	helpers.download(
		f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii',
		copy_to=os.path.join(DATA_DIR, 'surface', f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'))
	#> SJH 1012 parcellation
	helpers.download(f'https://github.com/MICA-MNI/micaopen/raw/master/MPC/maps/{hem.lower()}h.sjh.annot',f'{hem.lower()}h_sjh.annot')
	#> Von Economo regions
	helpers.download(f'https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/raw/master/fsaverage_fs6/label/{hem.lower()}h.economo.annot', f'{hem.lower()}h_economo.annot')
	#> Layers thicknesses (copy in data/surface)
	for layer_num in range(1,7):
		helpers.download(
			f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt',
			copy_to=os.path.join(DATA_DIR, 'surface', f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'))
