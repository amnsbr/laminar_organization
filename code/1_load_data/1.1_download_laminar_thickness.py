# Downloads BigBrain laminar thickness data from bigbrainwarp repository
import helpers
for hem in ['L', 'R']:
	for layer_num in range(1,7):
		helpers.download(
						f'https://github.com/caseypaquola/BigBrainWarp/raw/master/spaces/tpl-bigbrain/tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt',
						f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
						)
