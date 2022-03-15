import datasets
import helpers
import matrices
import surfaces

def create_matrices_and_gradients():
    for parcellation_name in ['sjh', 'schaefer400']:
        print("Parcellation: ", parcellation_name)
        for exc_adys in [True, False]:
            #> load/create curvature similarity matrix
            print("Loading/creating curvature similarity matrix")
            curvature_similarity_matrix_obj = matrices.CurvatureSimilarityMatrix(parcellation_name)
            #> load/create geodesic distance matrix
            print("Loading/creating geodesic distance matrix")
            geodesic_distance_matrix_obj = matrices.GeodesicDistanceMatrix(parcellation_name)
            #> load connectivity matrices
            if parcellation_name == 'schaefer400':
                sc_matrix_obj = matrices.ConnectivityMatrix(
                    'structural',
                    exc_adys=exc_adys,
                    parcellation_name=parcellation_name
                    )
                fc_matrix_obj = matrices.ConnectivityMatrix(
                    'functional',
                    exc_adys=exc_adys,
                    parcellation_name=parcellation_name
                    )
            #> load/create microstructure similarity matrices and gradients based on different options
            print("Loading/creating matrices & gradients")
            for input_type in ['thickness', 'density', 'thickness-density']:
                for correct_curvature in ['volume', 'regress', None]:
                    matrix_obj = matrices.MicrostructuralCovarianceMatrix(
                        input_type,
                        correct_curvature = correct_curvature,
                        exc_adys = exc_adys,
                        parcellation_name = parcellation_name
                    )
                    # TODO: add matrix associations
                    # TODO: add more loops for different gradient options
                    gradients_obj = surfaces.MicrostructuralCovarianceGradients(
                        matrix_obj,
                    )
                    # TODO: add gradient associations


def disease_gradients():
    for psych_only in [False, True]:
        #> create and plot the matrix
        dis_cov_matrix = matrices.DiseaseCovarianceMatrix(
            parcellation_name='aparc', 
            exc_regions=None, 
            psych_only=psych_only
        )
        #> create and plot the gradients
        dis_cov_gradients = surfaces.DiseaseCovarianceGradients(dis_cov_matrix)
        #> run the dominance analysis
        dis_cov_gradients.microstructure_dominance_analysis(n_perm=20, exc_adys=True)
        # TODO: maybe also repeat this without excluding adys from microstructural maps
        #> run the cross-validation on the R2 of the combined model
        dis_cov_gradients.microstructure_spatial_cross_validation()


def run():
    create_matrices_and_gradients()
    disease_gradients()

if __name__=='__main__':
    run()