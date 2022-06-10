import datasets
import helpers
import matrices
import surfaces
import os, glob, sys
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
        print(parc)
        ltc = matrices.MicrostructuralCovarianceMatrix('thickness', parc)
        ltc.create_or_load_surrogates()

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
    ltcg.correlate(exc_subtypes, x_columns=['LTC G1'])
    ltcg.correlate(inh_subtypes, x_columns=['LTC G1'])


def fig3():
    ## map of cortical types
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

def fig4():
    ltc = matrices.MicrostructuralCovarianceMatrix('thickness', 'aparc')
    ## LTC G1 ~ DisCov G1
    discov = matrices.DiseaseCovarianceMatrix(exc_regions=None)
    disg = surfaces.Gradients(discov, n_components_report=1)
    ltcg = surfaces.MicrostructuralCovarianceGradients(ltc)
    ltcg.correlate(disg, n_perm=1000, x_columns=['LTC G1'], y_columns=['DisCov G1'])

def run():
    # correlate_ltcg_ei_subtypes()
    # correlate_laminar_features_ei_subtypes()
    # fig3()
    fig4()
    # create_matrices_and_gradients()
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