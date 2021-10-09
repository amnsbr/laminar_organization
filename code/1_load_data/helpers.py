import urllib.request
import shutil
from urllib.parse import urlparse
import os
import numpy as np
import brainspace.mesh, brainspace.plotting

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', '..', 'data')


def download(url, file_name=None):
	"""
	Download the file from `url` and save it locally under `file_name`
	"""
	if not file_name:
		file_name = os.path.basename(urlparse(url).path)
	print(file_name, end=' ')
	if not os.path.exists(file_name):
		with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)
		print("")
	else:
		print("already exists")


def plot_on_bigbrain(surface_data_files, outfile=None):
	"""
	Plots the `surface_data_files` on the bigbrain space and saves it in `outfile`

	Note: Does not work in a remote server with proper X forwarding

	Parameters
	----------
	surface_data_files: (list of str) including paths to the surface file for L and R hemispheres
	outfile: (str) path to output; default would be the same as surface file
	"""
	#> load bigbrain surfaces
	lh_surf = brainspace.mesh.mesh_io.read_surface(
		os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-mid.surf.gii')
		)
	rh_surf = brainspace.mesh.mesh_io.read_surface(
		os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-mid.surf.gii')
		)
	#> read surface data files and concatenate L and R
	if surface_data_files[0].endswith('.npy'):
		surface_data = np.concatenate([np.load(surface_data_files[0]), np.load(surface_data_files[1])])
	else:
		print("Surface data file not supported")
		return
	if not outfile:
		outfile = surface_data_files[0]+'.png'
	brainspace.plotting.surface_plotting.plot_hemispheres(lh_surf, rh_surf, surface_data,
		color_bar=True, interactive=False, embed_nb=False, size=(1600, 400), zoom=1.2,
		screenshot=True, filename=outfile, transparent_bg=True, offscreen=True)
