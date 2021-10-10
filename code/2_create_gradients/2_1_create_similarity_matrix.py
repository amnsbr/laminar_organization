import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
import nilearn.surface

#> specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
DATA_DIR = os.path.join(cwd, '..', '..', 'data')


class LaminarSimilarityMatrix:
    def __init__(self, normalize=True, exc_masks=None, parcellation_name='sjh',
                 averaging_method='median', similarity_method='partial_corr', 
                 plot=False, filename=None):
        """
        Initializes laminar thickness similarity matrix object

        Parameters
        ---------
        normalize: (bool) Normalize by total thickness. Default: True
        exc_masks: (dict of str) Path to the surface masks of vertices that should be excluded (L and R)
        parcellation_name: (str) Parcellation scheme
            - 'sjh'
        averaging_method: (str) Method of averaging over vertices within a parcel. 
            - 'median'
            - 'mean'
        similarity_method: (str) 
            - 'partial_corr' -> partial correlation, 
            - 'pearson' -> Pearson's correlation
            - 'KL' -> KL-divergence of thickness distribution between parcels
        plot: (bool) Plot the similarity matrix. Default: False
        filenmae: (str) Output path to the figure of matrix heatmap
        """
        self.normalize = normalize
        self.exc_masks = exc_masks
        self.parcellation_name = parcellation_name
        self.averaging_method = averaging_method
        self.similarity_method = similarity_method

        self.create()
        self.plot()
        self.save()

    def create(self):
        """
        Creates laminar thickness similarity matrix
        """
        print("Reading laminar thickness files")
        self.laminar_thickness = self.read_laminar_thickness()
        if self.similarity_method != 'KL':
            print("Parcellating the data")
            self.parcellated_laminar_thickness = self.parcellate()
            print(f"Creating similarity matrix by {self.similarity_method}")
            self.matrix = self.create_by_corr()

    def read_laminar_thickness(self):
        """
        Reads laminar thickness data from 'data' folder and after masking out
        `exc_mask` returns 6-d laminar thickness arrays for left and right hemispheres

        Retruns
        --------
        laminar_thickness: (dict of np.ndarray) 6 x n_vertices for laminar thickness of L and R hemispheres
        """
        #> get the number of vertices
        n_vertices = np.loadtxt(os.path.join(
            DATA_DIR, 'surface',
            'tpl-bigbrain_hemi-L_desc-layer1_thickness.txt'
            )).size
        laminar_thickness = {}
        for hem in ['L', 'R']:
            #> read the laminar thickness data from bigbrainwrap .txt files
            laminar_thickness[hem] = np.empty((6, n_vertices))
            for layer_num in range(1, 7):
                laminar_thickness[hem][layer_num-1, :] = np.loadtxt(
                    os.path.join(
                        DATA_DIR, 'surface',
                        f'tpl-bigbrain_hemi-{hem}_desc-layer{layer_num}_thickness.txt'
                        ))
            #> remove the exc_mask
            if self.exc_masks:
                exc_mask_map = np.load(os.path.join(
                    DATA_DIR, 'surface',
                    self.exc_masks[hem]
                    ))
                laminar_thickness[hem][:, exc_mask_map] = np.NaN
        return laminar_thickness

    def parcellate(self):
        """
        Parcellates laminar thickness data using `parcellation` and by taking the
        median or mean (specified via `averaging_method`) of the vertices within each parcel.

        Returns
        ---------
        parcellated_laminar_thickness: (dict of pd.DataFrame) n_parcels x 6 for laminar thickness of L and R hemispheres
        """
        parcellated_laminar_thickness = {}
        for hem in ['L', 'R']:
            parcellation_map = nilearn.surface.load_surf_data(
                os.path.join(
                    DATA_DIR, 'parcellations', 
                    f'tpl-bigbrain_hemi-{hem}_desc-{self.parcellation_name}_parcellation.label.gii')
                )
            parellated_vertices = (
                pd.DataFrame(self.laminar_thickness[hem].T, index=parcellation_map)
                .reset_index()
                .groupby('index')
            )
            if self.averaging_method == 'median':
                parcellated_laminar_thickness[hem] = parellated_vertices.median()
            elif self.averaging_method == 'mean':
                parcellated_laminar_thickness[hem] = parellated_vertices.mean()
            
        return parcellated_laminar_thickness

    def create_by_corr(self):
        """
        Creates laminar thickness similarity matrix by taking partial or Pearson's correlation
        between pairs of parcels. In partial correlation the average laminar thickness pattern 
        is set as the covariate

        Note: Partial correlation is based on "Using recursive formula" subsection in the wikipedia
        entry for "Partial correlation", which is also the same as Formula 2 in Paquola et al. PBio 2019
        (https://doi.org/10.1371/journal.pbio.3000284)

        Returns
        -------
        matrix: (np.ndarray) n_parcels x n_parcels: how similar are each pair of parcels in their
                laminar thickness pattern
        """
        #> normalize by total thickness if needed
        if self.normalize:
            for hem in ['L', 'R']:
                self.parcellated_laminar_thickness[hem] =\
                    self.parcellated_laminar_thickness[hem].divide(
                        self.parcellated_laminar_thickness[hem].sum(axis=1), axis=0
                        )
        #> concatenate left and right hemispheres
        self.parcellated_laminar_thickness['R'].index = \
            self.parcellated_laminar_thickness['L'].index[-1] \
            + 1 \
            + self.parcellated_laminar_thickness['R'].index
        concat_parcellated_laminar_thickness = pd.concat(
            [
                self.parcellated_laminar_thickness['L'], 
                self.parcellated_laminar_thickness['R']
            ], 
            axis=0).dropna()
        #> create similarity matrix
        if self.similarity_method == 'partial_corr':
            #> calculate partial correlation
            r_ij = np.corrcoef(concat_parcellated_laminar_thickness)
            mean_laminar_thickness = concat_parcellated_laminar_thickness.mean()
            r_ic = concat_parcellated_laminar_thickness\
                        .corrwith(mean_laminar_thickness, 
                        axis=1) # r_ic and r_jc are the same
            r_icjc = np.outer(r_ic, r_ic) # the second r_ic is actually r_jc
            matrix = (r_ij - r_icjc) / np.sqrt(np.outer((1-r_ic**2),(1-r_ic**2)))
            #> zero out negative values
            matrix[matrix<0] = 0
            #> zero out correlations of 1 (to avoid division by 0)
            matrix[matrix==1] = 0
            #> Fisher's z-transformation
            matrix = 0.5 * np.log((1 + matrix) /  (1 - matrix))
            #> zero out NaNs and inf
            matrix[np.isnan(matrix) | np.isinf(matrix)] = 0
        elif self.similarity_method == 'pearson':
            matrix = np.corrcoef(concat_parcellated_laminar_thickness)
            #> zero out negative values
            matrix[matrix < 0] = 0
        return matrix
    
    def save(self, outfile=None):
        """
        Save the matrix to a .npy file
        """
        if not outfile:
            outfilename = f'matrix_method-{self.similarity_method}_parcellation-{self.parcellation_name}-{self.averaging_method}'
            if self.normalize:
                outfilename += '_normalized'
            if self.exc_masks: #TODO specify the name of excmask
                outfilename += '_excmask'
            outfilename = outfilename.lower()
            outfile = os.path.join(DATA_DIR, 'matrices', outfilename)
        np.save(outfile, self.matrix)

    def plot(self, outfile=None):
        """
        Plot the matrix as heatmap
        """
        fig, ax = plt.subplots(figsize=(7,7))
        heatmap(
            self.matrix,
            vmax=np.quantile(self.matrix.flatten(),0.75),
            cbar=False,
            ax=ax)
        ax.axis('off')
        if not outfile:
            outfilename = f'matrix_{self.similarity_method}_{self.parcellation_name}_parcellation_{self.averaging_method}'
            if self.normalize:
                outfilename += '_normalized'
            if self.exc_masks: #TODO specify the name of excmask
                outfilename += '_excmask'
            outfilename = outfilename.lower() + '.png'
            outfile = os.path.join(DATA_DIR, 'matrices', outfilename)
        fig.tight_layout()
        fig.savefig(outfile)

matrix = LaminarSimilarityMatrix()