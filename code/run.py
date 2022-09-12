import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import helpers
import matrices
import surfaces
import os, glob, sys
import statsmodels
import abagen

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# in progress

# specify the data dir
abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
OUTPUT_DIR = os.path.join(cwd, '..', 'output')
SRC_DIR = os.path.join(cwd, '..', 'src')

def create_matrices_and_gradients():
    for parcellation_name in ['sjh', 'schaefer400', 'mmp1', 'aparc', 'schaefer1000']:
        print("Parcellation: ", parcellation_name)
        for exc_regions in ['adysgranular', 'allocortex']:
            #> load/create curvature similarity matrix
            print("Loading/creating curvature similarity matrix")
            curvature_similarity_matrix_obj = matrices.CurvatureSimilarityMatrix(parcellation_name)
            #> load/create geodesic distance matrix
            print("Loading/creating geodesic distance matrix")
            geodesic_distance_matrix_obj = matrices.DistanceMatrix(parcellation_name)
            #> load connectivity matrices
            if 'schaefer' in parcellation_name:
                sc_matrix_obj = matrices.ConnectivityMatrix(
                    'structural',
                    exc_regions=exc_regions,
                    parcellation_name=parcellation_name
                    )
                fc_matrix_obj = matrices.ConnectivityMatrix(
                    'functional',
                    exc_regions=exc_regions,
                    parcellation_name=parcellation_name
                    )
            #> load/create microstructure similarity matrices and gradients based on different options
            print("Loading/creating matrices & gradients")
            for input_type in ['thickness', 'density', 'thickness-density']:
                for correct_curvature in ['volume', 'regress', None]:
                    matrix_obj = matrices.MicrostructuralCovarianceMatrix(
                        input_type,
                        correct_curvature = correct_curvature,
                        exc_regions = exc_regions,
                        parcellation_name = parcellation_name
                    )
                    # TODO: add matrix associations
                    # TODO: add more loops for different gradient options
                    gradients_obj = surfaces.MicrostructuralCovarianceGradients(
                        matrix_obj,
                    )
                    gradients_obj.plot_surface()
                    # TODO: add gradient associations


def disease_gradients_analyses():
    for psych_only in [False, True]:
        #> create and plot the matrix
        dis_cov_matrix = matrices.DiseaseCovarianceMatrix(
            parcellation_name='aparc', 
            exc_regions=None, 
            psych_only=psych_only
        )
        #> create and plot the gradients
        dis_cov_gradients = surfaces.Gradients(dis_cov_matrix)
        #> run the dominance analysis
        dis_cov_gradients.microstructure_dominance_analysis(col_idx=0, n_perm=20, exc_adys=True)

def ei_analyses():
    for receptor in ['NMDA', 'GABAa']:
        #> load the receptor map
        receptor_map = surfaces.PETMaps(receptor, 'sjh')
        #> plot it
        helpers.plot_on_bigbrain_nl(
            receptor_map.surf_data[:, 0],
            receptor_map.file_path.replace('.csv','.png'),
            inflate=True,
            plot_downsampled=True,
            # cmap=???
        )
        #> association with microstructure
        receptor_map.microstructure_dominance_analysis(col_idx=0, n_perm=1000, exc_adys=True)

def create_surrogates():
    for parc in ['aparc', 'schaefer400', 'mmp1', 'sjh']:
    #     print(parc)
    #     ltc = matrices.MicrostructuralCovarianceMatrix('thickness', parc)
    #     ltc.create_or_load_surrogates()
        helpers.get_rotated_parcels(parc, n_perm=1000)

def gradients_robustness():
    default_ltcg = surfaces.MicrostructuralCovarianceGradients(
        matrices.MicrostructuralCovarianceMatrix('thickness')
    )
    res_dir = os.path.join(OUTPUT_DIR, 'ltc', 'gradient_robustness')
    os.makedirs(res_dir, exist_ok=True)
    res_str = ""
    # parcellations
    for parc in [None, 'schaefer1000', 'schaefer400', 'aparc', 'mmp1']:
        res_str += f"Parcellation: {parc}\n"
        curr_ltcg = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('thickness', parcellation_name=parc),
            n_components_report = 1, create_plots=True
        )
        coefs, pvals = curr_ltcg.correlate(default_ltcg, parcellated=False, x_columns=['LTC G1'], y_columns=['LTC G1'])
        res_str += f'Coefficient: {coefs.iloc[0, 0]}, p-value: {pvals.iloc[0, 0]}\n---\n'
    # mask
    res_str += "Not removing a-/dysgranular regions"
    curr_ltcg = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('thickness', exc_regions=None),
            n_components_report = 1, create_plots = True
        )
    coefs, pvals = curr_ltcg.correlate(default_ltcg, parcellated=False, x_columns=['LTC G1'], y_columns=['LTC G1'])
    res_str += f'Coefficient: {coefs.iloc[0, 0]}, p-value: {pvals.iloc[0, 0]}\n---\n'
    # LTC metric
    for metric in ['pearson', 'euclidean']:
        res_str += f"LTC metric: {metric}\n"
        curr_ltcg = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('thickness', similarity_metric=metric),
            n_components_report = 1, create_plots = True
        )
        coefs, pvals = curr_ltcg.correlate(default_ltcg, parcellated=False, x_columns=['LTC G1'], y_columns=['LTC G1'])
        res_str += f'Coefficient: {coefs.iloc[0, 0]}, p-value: {pvals.iloc[0, 0]}\n---\n'
    # gradient approach
    for approach in ['pca', 'le']:
        res_str += f"Gradient approach: {approach}\n"
        curr_ltcg = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('thickness'),
            approach = approach,
            n_components_report = 1, create_plots = True
        )
        coefs, pvals = curr_ltcg.correlate(default_ltcg, parcellated=False, x_columns=['LTC G1'], y_columns=['LTC G1'])
        res_str += f'Coefficient: {coefs.iloc[0, 0]}, p-value: {pvals.iloc[0, 0]}\n---\n'
    with open(os.path.join(res_dir, 'gradient_robustness_summary.txt'), 'w') as res_file:
        res_file.write(res_str)
    sparsity
    ltcg_per_sparsity = {}
    for sparsity in np.arange(0, 1, 0.1):
        sparsity = round(sparsity, 2)
        ltcg_per_sparsity[f'Sparsity_{sparsity}'] = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('thickness'),
            sparsity = sparsity,
            n_components_report = 1,
            create_plots = False
        ).surf_data[:, :1]
    ltcg_per_sparsity_surf_obj = surfaces.ContCorticalSurface(
        surf_data = np.hstack(ltcg_per_sparsity.values()),
        columns = list(ltcg_per_sparsity.keys()),
        label = 'LTCG with variable sparsity',
        dir_path = res_dir,
    )
    coefs, pvals = ltcg_per_sparsity_surf_obj.correlate(ltcg_per_sparsity_surf_obj, parcellated = False)
    coefs.values[np.triu_indices_from(coefs, 1)] = np.NaN
    coefs.index = coefs.columns = range(0, 100, 10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(coefs, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('% Sparsity')
    ax.set_ylabel('% Sparsity')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, 'coefs.png'))

def fused_gradient():
    fused = matrices.MicrostructuralCovarianceMatrix('thickness-density', laminar_density=True)
    fusedg = surfaces.MicrostructuralCovarianceGradients(fused, create_plots = False, n_components_report = 1)
    ltcg = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('thickness')
        )
    mpcg = surfaces.MicrostructuralCovarianceGradients(
            matrices.MicrostructuralCovarianceMatrix('density')
        )
    fusedg.correlate(ltcg, parcellated=True, x_columns=['Fused G1'], y_columns=['LTC G1'])
    fusedg.correlate(mpcg, parcellated=True, x_columns=['Fused G1'], y_columns=['MPC G1'])
    fusedg.correlate(ltcg, parcellated=True, x_columns=['Fused G2'], y_columns=['LTC G2'])
    fusedg.correlate(mpcg, parcellated=True, x_columns=['Fused G2'], y_columns=['MPC G2'])

def correlate_laminar_features_ei_subtypes(parcellated=False):
    laminar_features = surfaces.LaminarFeatures('sjh', exc_regions='adysgranular')
    exc_subtypes = matrices.NeuronalSubtypesCovarianceMatrix('exc', 'sjh').get_surface()
    inh_subtypes = matrices.NeuronalSubtypesCovarianceMatrix('inh', 'sjh').get_surface()
    selected_columns = [f'Layer {num} absolute thickness' for num in range(1, 7)] +\
        [f'Layer {num} density' for num in range(1, 7)]
    laminar_features.correlate(exc_subtypes, parcellated=parcellated, x_columns=selected_columns)
    laminar_features.correlate(inh_subtypes, parcellated=parcellated, x_columns=selected_columns)

def correlate_ltcg_ei_subtypes():
    ltc = matrices.MicrostructuralCovarianceMatrix('thickness', 'sjh', create_plots=False)
    ltcg = surfaces.MicrostructuralCovarianceGradients(ltc)
    exc_subtypes = matrices.NeuronalSubtypesCovarianceMatrix('exc', 'sjh').get_surface()
    inh_subtypes = matrices.NeuronalSubtypesCovarianceMatrix('inh', 'sjh').get_surface()
    ltcg.correlate(exc_subtypes, x_columns=['LTC G1'], sort_barplot=False)
    ltcg.correlate(inh_subtypes, x_columns=['LTC G1'], sort_barplot=False)


def fig3():
    # panel A
    # panel B
    ctypes = surfaces.CorticalTypes(exc_regions=None, downsampled=False)
    # helpers.plot_surface(
    #     ctypes.surf_data,
    #     os.path.join(ctypes.dir_path, 'surface_grid'),
    #     inflate=True,
    #     plot_downsampled=False,
    #     layout_style='grid',
    #     cmap='RdYlGn_r',
    # )
    ## comparison of LTC and cortical types
    ltc = matrices.MicrostructuralCovarianceMatrix('thickness', 'sjh', create_plots=False)
    # ltc.associate_cortical_types(stats=True)
    ## comparison of LTC G1 and cortical types
    ltcg = surfaces.MicrostructuralCovarianceGradients(ltc)
    ctypes.compare(ltcg, ['LTC G1'])
    # panel C
    mri_mpcg = surfaces.MriMpcGradients()
    ltcg = surfaces.MicrostructuralCovarianceGradients(
        matrices.MicrostructuralCovarianceMatrix('thickness', 'schaefer400'))
    # ltcg.correlate(mri_mpcg, x_columns=['LTC G1'], y_columns=['MPCmri G1'], parcellated=True, n_perm=1000)
    mri_mpcg.correlate(ltcg, y_columns=['LTC G1'], x_columns=['MPCmri G1'], parcellated=True, n_perm=1000)

def fig4():
    # ## FC and SC likelihood analyses
    # sc = matrices.ConnectivityMatrix('structural')
    # fc = matrices.ConnectivityMatrix('functional')
    # gd = matrices.DistanceMatrix('schaefer400')
    ltc = matrices.MicrostructuralCovarianceMatrix('thickness', 'schaefer400')
    # ctype_diff = matrices.CorticalTypeDiffMatrix('schaefer400', 'adysgranular')
    # fc.binarized_association([gd, ltc, ctype_diff])
    # sc.binarized_association([gd, ltc, ctype_diff])
    ## EC
    ec_maps = surfaces.EffectiveConnectivityMaps(dataset='mics')
    ec_maps.surf_data = ec_maps.surf_data.astype('float64')
    ltcg = surfaces.MicrostructuralCovarianceGradients(ltc)
    ltcg.correlate(ec_maps, x_columns=['LTC G1'], n_perm=1000, axis_off=True)
    # myelin_map = surfaces.MyelinMap('schaefer400', exc_regions='adysgranular')
    # myelin_map.correlate(ec_maps, n_perm=1000, axis_off=True)
    # ctypes = surfaces.CorticalTypes(exc_regions='adysgranular', parcellation_name='schaefer400')
    # ctypes.compare(ec_maps)


def fig5():
    ltc = matrices.MicrostructuralCovarianceMatrix('thickness', 'aparc')
    ## LTC G1 ~ DisCov G1
    discov = matrices.DiseaseCovarianceMatrix(exc_regions=None)
    disg = surfaces.Gradients(discov, n_components_report=1)
    ltcg = surfaces.MicrostructuralCovarianceGradients(ltc)
    ltcg.correlate(disg, n_perm=1000, x_columns=['LTC G1'], y_columns=['DisCov G1'])

def brainmap_transdiagnostic_ale():
    import nimare
    for modality in ['vbp', 'vbm']:
        disease_vbm_dset = nimare.io.convert_sleuth_to_dataset(
            os.path.join(OUTPUT_DIR, f'brainmap_disease_normal_{modality}.txt')
        )
        ma_maps = nimare.meta.kernel.ALEKernel().transform(
            disease_vbm_dset, disease_vbm_dset.masker, return_type='array'
            )
        ale_map = nimare.meta.cbma.ale.ALE()._compute_summarystat(ma_maps)
        ale_img = disease_vbm_dset.masker.inverse_transform(ale_map)
        ale_img.to_filename(f'brainmap_disease_normal_{modality}_ale.nii.gz')

def ahba_ltcg1_correlation(brain_specific=True, fdr=True):
    ahba_df = datasets.fetch_ahba_data('sjh', ibf_threshold=0.5, missing='centroids')['all']
    if brain_specific:
        brain_specific_genes_in_ahba = ahba_df.columns.intersection(abagen.fetch_gene_group('brain'))
        ahba_df = ahba_df.loc[:, brain_specific_genes_in_ahba]
    ltcg = surfaces.MicrostructuralCovarianceGradients(matrices.MicrostructuralCovarianceMatrix('thickness', 'sjh'))
    ltcg1 = ltcg.parcellated_data.iloc[:, :1]
    coefs, pvals, null_dist = helpers.variogram_test(
        ltcg1, 
        ahba_df, 
        'sjh',
        surrogates_path=os.path.join(ltcg.dir_path, 'variogram_surrogates_LTC G1')
    )
    _, corr_pvals = statsmodels.stats.multitest.fdrcorrection(pvals.values[:, 0])
    res = pd.concat([coefs, pvals], axis=1)
    res.columns = ['coef', 'pval']
    res.loc[:,'pval_fdr'] = corr_pvals
    res.to_csv(os.path.join(
        ltcg.dir_path, 
        'ahba_correlation_variogram' + ('_brain.csv' if brain_specific else '.csv'))
        )
    if fdr:
        sig_res = res.loc[res['pval_fdr'] < 0.05]
    else:
        sig_res = res.loc[res['pval'] < 0.05]
    for direction in ['pos', 'neg']:
        if direction == 'neg':
            direction_sig_genes = sig_res.loc[sig_res['coef'] < 0].index.tolist()
        else:
            direction_sig_genes = sig_res.loc[sig_res['coef'] >= 0].index.tolist()

        with open(os.path.join(
            ltcg.dir_path, 
            'ahba_correlation_variogram' + ('_brain' if brain_specific else '') + ('_fdr' if fdr else '') + f'_{direction}.txt'
            ), 'w') as f:
            f.write('\n'.join(direction_sig_genes))


def ahba_ltcg1_pls(n_perm=1000):
    from sklearn.cross_decomposition import PLSRegression
    ahba_df = datasets.fetch_ahba_data('sjh', ibf_threshold=0.25, missing='centroids')['all']
    ltcg = surfaces.MicrostructuralCovarianceGradients(matrices.MicrostructuralCovarianceMatrix('thickness', 'sjh'))
    ltcg1 = ltcg.parcellated_data.iloc[:, :1]
    # get the original weights
    shared_parcels = ahba_df.index.intersection(ltcg1.index)
    x = ahba_df.loc[shared_parcels]
    y_orig = ltcg1.loc[shared_parcels]
    pls_orig = PLSRegression(n_components=1)
    pls_orig.fit(x, y_orig)
    weights_orig=pls_orig.x_weights_[:, 0]    
    # load ltc g1 surrogates
    surrogates_path=os.path.join(ltcg.dir_path, 'variogram_surrogates_LTC G1_nperm-1000_nparcels-444.npz')
    surrogates = np.load(surrogates_path)['surrogates']
    parcels = np.load(surrogates_path)['parcels']
    assert (parcels == shared_parcels).all()
    # get the null distribution of weights
    weights_null = np.zeros((n_perm, ahba_df.shape[1]))
    for i in range(n_perm):
        pls_surrogate = PLSRegression(n_components=1)
        pls_surrogate.fit(x, surrogates[i, :, :])
        weights_null[i, :] = pls_surrogate.x_weights_[:, 0]
    np.savez_compressed('pls_res.npz', 
        weights_orig=weights_orig,
        weights_null=weights_null,
        gene_names=ahba_df.columns.to_list()
        )

def create_downsampled_mpc():
    mpc = matrices.MicrostructuralCovarianceMatrix('density', parcellation_name=None, exc_regions=None)

def brainspan_analyses():
    brainspan = pd.read_csv(os.path.join(SRC_DIR,'brainspan/expression_matrix.csv'), index_col=0, header=None)
    columns_metadata = pd.read_csv(os.path.join(SRC_DIR,'brainspan/columns_metadata.csv'), index_col=0)
    rows_metadata = pd.read_csv(os.path.join(SRC_DIR,'brainspan/rows_metadata.csv'), index_col=0)
    brainspan.index = rows_metadata['gene_symbol']
    age_groups = [
        ['8 pcw', '9 pcw'],
        ['12 pcw'],
        ['13 pcw'],
        ['16 pcw', '17 pcw'],
        ['19 pcw', '21 pcw', '24 pcw'],
        ['25 pcw', '26 pcw', '35 pcw', '37 pcw'],
        ['4 mos'],
        ['10 mos', '1 yrs'],
        ['2 yrs', '3 yrs', '4 yrs'],
        ['8 yrs', '11 yrs'],
        ['13 yrs', '15 yrs', '18 yrs', '19 yrs'],
        ['21 yrs', '23 yrs', '30 yrs', '36 yrs', '37 yrs', '40 yrs']
    ]
    brainspan_brodmann_mapping = {
        'V1C': ['BA17'],
        'A1C': ['BA41_42_52'],
        'S1C': ['BA1_3'],
        'M1C': ['BA4'],
        'STC': ['BA22'],
        'ITC': ['BA20', 'BA21'],
        'IPC': ['BA39', 'BA40'],
        'DFC': ['BA9', 'BA46'],
        'VFC': ['BA44', 'BA45', 'BA47'],
        'OFC': ['BA10', 'BA11'],
        'MFC': ['BA24', 'BA32', 'BA33']
    }
    ltcg_brodmann = surfaces.MicrostructuralCovarianceGradients(matrices.MicrostructuralCovarianceMatrix('thickness', 'brodmann'), create_plots=False)
    ltcg_brainspan_regions = pd.Series()
    for brainspan_region, brodmann_parcels in brainspan_brodmann_mapping.items():
        try:
            ltcg_brainspan_regions.loc[brainspan_region] = \
                ltcg_brodmann.parcellated_data.loc[brodmann_parcels, 'LTC G1'].mean()
        except KeyError:
            pass
    gene_list = list(set(abagen.fetch_gene_group('brain')) & set(brainspan.index))
    regional_developmental_similarity = np.zeros((len(gene_list), 10, 10))
    for gene_idx, gene in enumerate(gene_list):
        gene_exp_by_age = pd.DataFrame()
        for struc in ltcg_brainspan_regions.index:
            for i in range(len(age_groups)):
                if len(age_groups[i]) == 1:
                    age_str = age_groups[i][0]
                else:
                    age_str = f'{age_groups[i][0]}-{age_groups[i][-1]}'

                samples = columns_metadata[
                    (columns_metadata['structure_acronym']==struc) &\
                    (columns_metadata['age'].isin(age_groups[i]))
                ].index
                if samples.size > 0:
                    gene_exp_by_age.loc[struc, age_str] = brainspan.loc[gene, samples].values.mean()
        regional_developmental_similarity[gene_idx, :, :] = gene_exp_by_age.T.corr().values
    np.savez_compressed('regional_developmental_similarity.npz', 
    regional_developmental_similarity=regional_developmental_similarity,
    gene_list = gene_list)

def regional_ltc():
    ltc = matrices.MicrostructuralCovarianceMatrix('thickness', None)
    ltc._load()
    vmin = np.nanquantile(ltc.matrix.values, 0.1)
    vmax = np.nanquantile(ltc.matrix.values, 0.9)
    out_dir = os.path.join(ltc.dir_path, 'regional_ltc')
    os.makedirs(out_dir, exist_ok=True)
    brodmann_lh_parcel_centers = helpers.get_parcel_center_indices('brodmann', downsampled=True)['L']
    for roi, center_vertex in brodmann_lh_parcel_centers.items():
        print(roi)
        file_path = os.path.join(out_dir, roi)
        center_vertex = brodmann_lh_parcel_centers.loc[roi]
        if not center_vertex in ltc.matrix.index:
            print("LTC for the region is not defined")
            continue
        # get nodal ltc (while taking care of removed vertices)
        nodal_ltc = pd.concat([ltc.matrix.loc[center_vertex], pd.Series(0, index=np.arange(datasets.N_VERTICES_HEM_BB_ICO5*2))], axis=1).iloc[:, 0]
        nodal_ltc = helpers.upsample(nodal_ltc.values)
        helpers.plot_surface(nodal_ltc, file_path, inflate=False, plot_downsampled=False, vrange=(vmin, vmax), cbar=True)

def run():
    pass
    # regional_ltc()
    # brainspan_analyses()
    # ec_maps = surfaces.EffectiveConnectivityMaps(dataset='mics')
    # exc_cov = matrices.NeuronalSubtypesCovarianceMatrix('exc', 'schaefer400', exc_regions=None)
    # exc_subtypes_maps = exc_cov.get_surface()
    # ec_maps.correlate(exc_subtypes_maps)
    # surfaces.InhMarkers(parcellation_name='schaefer400')
    # ahba_ltcg1_correlation(brain_specific=False)
    # ahba_ltcg1_pls()
    # create_downsampled_mpc()
    # brainmap_transdiagnostic_ale()
    # correlate_ltcg_ei_subtypes()
    # correlate_laminar_features_ei_subtypes()
    # fig3()
    # fig4()
    # create_matrices_and_gradients()
    # gradients_robustness()
    fused_gradient()
    # disease_gradients_analyses()
    # ei_analyses()
    # create_surrogates()
    # for correct_curvature in ['smooth-10', None]:
    # ltc = matrices.MicrostructuralCovarianceMatrix('thickness', 'sjh')
    # ltcg = surfaces.MicrostructuralCovarianceGradients(ltc)
    # mpc = matrices.MicrostructuralCovarianceMatrix('density', 'sjh')
    # mpcg = surfaces.MicrostructuralCovarianceGradients(mpc)
    # ltcg.correlate(mpcg, x_columns=['LTC G1'], y_columns=['MPC G1'])
    # ec_maps = surfaces.EffectiveConnectivityMaps(dataset='mics')
    # ltcg = surfaces.MicrostructuralCovarianceGradients(
    #     matrices.MicrostructuralCovarianceMatrix('thickness', 'schaefer400')
    # )
    # ltcg.correlate(ec_maps, x_columns=['LTC G1'], axis_off=True)
    # nmda = surfaces.PETMaps('NMDA', 'sjh')
    # ltcg.correlate(nmda, x_columns=['LTC G1'], axis_off=True)
    # gaba = surfaces.PETMaps('GABAa', 'sjh')
    # ltcg.correlate(gaba, x_columns=['LTC G1'], axis_off=True)
    # correlate_laminar_features_ei_subtypes(parcellated=False)    

if __name__=='__main__':
    run()