"""
Downloads BigBrain laminar thickness data from bigbrainwarp repository
"""
import os
import helpers

abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
SRC_PATH = os.path.join(cwd, '..', '..', 'src')
os.chdir(SRC_PATH)

for hem in ['L', 'R']:
	helpers.download(
					f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii',
					f'tpl-bigbrain_hemi-{hem}_desc-mid.surf.gii'
					)
	helpers.download(
					f'https://github.com/MICA-MNI/micaopen/raw/master/MPC/maps/{hem.lower()}h.sjh.annot',
					f'{hem.lower()}h.sjh.annot'
	)
	for layer_num in range(1,7):
		helpers.download(
						f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt',
						f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
						)
