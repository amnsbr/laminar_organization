"""
Downloads BigBrain laminar thickness data from bigbrainwarp repository
"""
from genericpath import exists
import os
import helpers

#> Create and change path to src folder
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
SRC_PATH = os.path.join(cwd, '..', '..', 'src')
os.makedirs(SRC_PATH, exist_ok=True)
os.chdir(SRC_PATH)


for hem in ['L', 'R']:
	#> Bigbrain space
	helpers.download(f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii')
	#> SJH 1012 parcellation
	helpers.download(f'https://github.com/MICA-MNI/micaopen/raw/master/MPC/maps/{hem.lower()}h.sjh.annot',f'{hem.lower()}h_sjh.annot')
	#> Von Economo regions
	helpers.download(f'https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/raw/master/fsaverage_fs6/label/{hem.lower()}h.economo.annot', f'{hem.lower()}h_economo.annot')
	#> Layers thicknesses
	for layer_num in range(1,7):
		helpers.download(f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt')