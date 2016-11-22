
# Simultaneous Perturbation Stochastic Approximation

# In this file, I'm suggesting that you implement the simultaneous perturbation estimator for the gradient, as
# documented in equation 6.6 of Spall, http://www.jhuapl.edu/spsa/PDF-SPSA/Handbook04_StochasticOptimization.pdf.
# As noted in the README.txt and in main.py

from tuple_operations import vector_sum, vector_difference


def validate(gradient_list):
    for gradient_tuple in gradient_list:
        for tuple_component in gradient_tuple:
            assert abs(tuple_component) < 100


# This is the implementation of Spall's equation 6.6.
def simultaneous_perturbation(optimization_objective, x_vector, theta_vector, x_perturbation, theta_perturbation):
    optimization_objective_plus = optimization_objective(vector_sum(x_vector, x_perturbation),
                                                         vector_sum(theta_vector, theta_perturbation))
    optimization_objective_minus = optimization_objective(vector_difference(x_vector, x_perturbation),
                                                          vector_difference(theta_vector, theta_perturbation))

    # This is what Spall, has in equation 6.6, in its last form. This is what appears out front.
    prefactor = (optimization_objective_plus - optimization_objective_minus) / 2.0

    gradient_x = [tuple([prefactor / tuple_component for tuple_component in x]) for x in x_perturbation]
    gradient_theta = [tuple([prefactor / tuple_component for tuple_component in theta]) for theta in theta_perturbation]

    validate(gradient_x)
    validate(gradient_theta)

    return gradient_x, gradient_theta
