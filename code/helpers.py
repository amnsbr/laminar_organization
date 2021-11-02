import urllib.request
import shutil
from urllib.parse import urlparse
import os
import numpy as np
import matplotlib.pyplot as plt
import brainspace.mesh, brainspace.plotting
import nilearn.surface
import nilearn.plotting

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', 'data')


def download(url, file_name=None, copy_to=None):
	"""
	Download the file from `url` and save it locally under `file_name`.
	Also creates a copy in 'copy_to'
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
	if copy_to:
		if not os.path.exists(copy_to):
			shutil.copyfile(file_name, copy_to)


def plot_on_bigbrain_brainspace(surface_data_files, outfile=None):
	"""
	Plots the `surface_data_files` on the bigbrain space and saves it in `outfile`
	using brainsapce.

	Note: Does not work in a remote server without proper X forwarding

	Parameters
	----------
	surface_data_files: (dict of str) including paths to the surface file for 'L' and 'R' hemispheres
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
	elif surface_data_files[0].endswith('.txt'):
		surface_data = np.concatenate([np.loadtxt(surface_data_files[0]), np.loadtxt(surface_data_files[1])])
	else:
		print("Surface data file not supported")
		return
	if not outfile:
		outfile = surface_data_files[0]+'.png'
	brainspace.plotting.surface_plotting.plot_hemispheres(lh_surf, rh_surf, surface_data,
		color_bar=True, interactive=False, embed_nb=False, size=(1600, 400), zoom=1.2,
		screenshot=True, filename=outfile, transparent_bg=True, offscreen=True)

def plot_on_bigbrain_nl(surface_data_files, outfile=None):
	"""
	Plots the `surface_data_files` on the bigbrain space and saves it in `outfile`
	using nilearn

	Parameters
	----------
	surface_data_files: (dict of str) including paths to the surface file for 'L' and 'R' hemispheres
	outfile: (str) path to output; default would be the same as surface file
	"""
	#> initialize the figures
	figure, axes = plt.subplots(1, 4, figsize=(24, 5), subplot_kw={'projection': '3d'})
	curr_ax_idx = 0
	for hem_idx, hemi in enumerate(['left', 'right']):
		#> read surface data files
		if surface_data_files[0].endswith('.npy'):
			surface_data = np.load(surface_data_files[hem_idx])
		elif surface_data_files[0].endswith('.txt'):
			surface_data = np.loadtxt(surface_data_files[hem_idx])
		else:
			print("Surface data file not supported")
			return
		#> plot the medial and lateral views
		if hemi == 'left':
			views_order = ['lateral', 'medial']
			mesh_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-L_desc-mid.surf.gii')
		else:
			views_order = ['medial', 'lateral']
			mesh_path = os.path.join(DATA_DIR, 'surface', 'tpl-bigbrain_hemi-R_desc-mid.surf.gii')
		for view in views_order:
			nilearn.plotting.plot_surf(
				mesh_path,
				surface_data,
				hemi=hemi, view=view, axes=axes[curr_ax_idx],
			)
			curr_ax_idx += 1
	figure.subplots_adjust(wspace=0, hspace=0)
	figure.tight_layout()
	if not outfile:
		outfile = surface_data_files[0]+'.png'
	figure.savefig(outfile, dpi=192)
