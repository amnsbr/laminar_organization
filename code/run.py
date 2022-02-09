import datasets
import helpers
import matrices
import gradients

def run():
    for parcellation_name in ['sjh', 'schaefer400']:
        print("Parcellation: ", parcellation_name)
        #> load/create a-/dysgranular mask
        print("Loading/creating a-/dysgranular masks")
        adysgranular_masks = datasets.load_adysgranular_masks(parcellation_name)
        #> load/create curvature similarity matrix
        print("Loading/creating curvature similarity matrix")
        curvature_similarity_matrix_obj = matrices.CurvatureSimilarityMatrix(parcellation_name)
        #> load/create geodesic distance matrix
        print("Loading/creating geodesic distance matrix")
        geodesic_distance_matrix_obj = matrices.GeodesicDistanceMatrix(parcellation_name)
        #> load/create microstructure similarity matrices and gradients based on different options
        print("Loading/creating matrices & gradients")
        for input_type in ['thickness', 'thickness-density', 'density']:
            for exc_masks in [adysgranular_masks, None]:
                for correct_curvature in ['volume', 'regress', None]:
                    matrix_obj = matrices.MicrostructuralCovarianceMatrix(
                        input_type,
                        correct_curvature = correct_curvature,
                        exc_masks = exc_masks,
                        parcellation_name = parcellation_name
                    )
                    # TODO: add matrix associations
                    # TODO: add more loops for different gradient options
                    gradients_obj = gradients.MicrostructuralCovarianceGradients(
                        matrix_obj
                    )
                    # TODO: add gradient associations

if __name__=='__main__':
    run()