"""
Downloads parcellations, spaces and laminar thickness data into 'src' folder
and copies those that need no preprocessing to the 'data' folder
"""
import os
from distutils.dir_util import copy_tree
import gzip
import shutil
import time
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

#> Download external sources
# Bigbrain volume (Warning: 1.4 GB)
helpers.download('https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Volumes/Histological_Space/mnc/full16_100um_optbal.mnc')
# helpers.download_bigbrain_ftp(
# 	'BigBrainRelease.2015/3D_Volumes/Histological_Space/mnc/',
# 	'full16_100um_optbal.mnc')
for hem in ['L', 'R']:
	# Bigbrain space (copy in data/)
	helpers.download(
		f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii',
		copy_to=os.path.join(DATA_DIR, 'surface', f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'))
	# SJH 1012 parcellation
	helpers.download(f'https://github.com/MICA-MNI/micaopen/raw/master/MPC/maps/{hem.lower()}h.sjh.annot',f'{hem.lower()}h_sjh.annot')
	# Von Economo regions
	helpers.download(f'https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/raw/master/fsaverage_fs6/label/{hem.lower()}h.economo.annot', f'{hem.lower()}h_economo.annot')
	# Layers thicknesses (copy in data/surface)
	for layer_num in range(1,7):
		helpers.download(
			f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt',
			copy_to=os.path.join(DATA_DIR, 'surface', f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'))
	# Layer boundaries
	for boundary_num in range(0, 7):
		filename = f"layer{boundary_num}_{hem.replace('L','left').replace('R','right')}_327680.obj.gz"
		#> Sometimes gzip files downloaded from bigbrain-ftp are incomplete
		#  The code below is for fixing this
		n_retries = 0
		while n_retries < 100:
			try:
				with gzip.open(filename, 'rb') as f_in: #unzip
					with open(filename.replace('.gz',''), 'wb') as f_out:
						shutil.copyfileobj(f_in, f_out)
			except:
				helpers.download(f'https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/Layer_Segmentation/3D_Surfaces/PLoSBiology2020/MNI-obj/{filename}', 
								 overwrite=True)
				n_retries += 1
				time.sleep(1)
			else:
				break

#> Copy local sources to src
copy_tree(os.path.join(SRC_DIR, '..', 'src_local'), SRC_DIR)