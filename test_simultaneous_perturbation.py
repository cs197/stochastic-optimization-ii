
# In this file, we test some tuple operations.

import unittest

from simultaneous_perturbation import simultaneous_perturbation

from tuple_operations import vector_inner_product, vector_sum, vector_difference

x_vector_for_testing = [(3, 5), (1, 2), (3, 4)]
theta_vector_for_testing = [(5, 7), (9, 10)]

x_perturbation_for_testing = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
theta_perturbation_for_testing = [(0.3, 0.4), (0.5, 0.6)]


def optimization_objective_for_testing(x_vector, theta_vector):
    # A dirt-simple function
    return vector_inner_product(x_vector, x_vector) + vector_inner_product(theta_vector, theta_vector)

# The following should be optimization_objective_function_for_testing evaluated at
# [(3.1, 5.2), (1.3, 2.4), (3.5, 4.6)], [(5.3, 7.4), (9.5, 10.6)]
# minus the same function evaluated at
# [(2.9, 4.8), (0.7, 1.6), (2.5, 3.4)], [(4.7, 6.6), (8.5, 9.4)]
# divided by 2.

# E.g.
# 3.1 * 3.1 + 5.2 * 5.2 + 1.3 * 1.3 + 2.4 * 2.4 + 3.5 * 3.5 + 4.6 * 4.6
# + 5.3 * 5.3 + 7.4 * 7.4 + 9.5 * 9.5 + 10.6 * 10.6
# minus
# 2.9 * 2.9 + 4.8 * 4.8 + 0.7 * 0.7 + 1.6 * 1.6 + 2.5 * 2.5 + 3.4 * 3.4
# + 4.7 * 4.7 + 6.6 * 6.6 + 8.5 + 8.5 + 9.4 + 9.4
# divided by 2

f_plus = 362.97

f_minus = 278.57

prefactor = (f_plus - f_minus) / 2


class TestSimultaneousPerturbation(unittest.TestCase):

    def test_optimization_function_for_testing(self):
        perturbed_x_vector = vector_sum(x_vector_for_testing, x_perturbation_for_testing)
        perturbed_theta_vector = vector_sum(theta_vector_for_testing, theta_perturbation_for_testing)
        self.assertAlmostEqual(optimization_objective_for_testing(perturbed_x_vector, perturbed_theta_vector),
                               f_plus, 7)

    def test_optimization_function_for_testing2(self):
        perturbed_x_vector = vector_difference(x_vector_for_testing, x_perturbation_for_testing)
        perturbed_theta_vector = vector_difference(theta_vector_for_testing, theta_perturbation_for_testing)
        self.assertAlmostEqual(optimization_objective_for_testing(perturbed_x_vector, perturbed_theta_vector),
                               f_minus, 7)

    def test_simultaneous_perturbation(self):
        result_x, result_theta = simultaneous_perturbation(optimization_objective_for_testing,
                                                           x_vector_for_testing, theta_vector_for_testing,
                                                           x_perturbation_for_testing, theta_perturbation_for_testing)

        # First check the x-direction gradients
        self.assertAlmostEqual(result_x[0][0], prefactor / x_perturbation_for_testing[0][0], 7)
        self.assertAlmostEqual(result_x[1][0], prefactor / x_perturbation_for_testing[1][0], 7)
        self.assertAlmostEqual(result_x[2][0], prefactor / x_perturbation_for_testing[2][0], 7)
        self.assertAlmostEqual(result_x[0][1], prefactor / x_perturbation_for_testing[0][1], 7)
        self.assertAlmostEqual(result_x[1][1], prefactor / x_perturbation_for_testing[1][1], 7)
        self.assertAlmostEqual(result_x[2][1], prefactor / x_perturbation_for_testing[2][1], 7)

        # Then check the theta-direction gradients
        self.assertAlmostEqual(result_theta[0][0], prefactor / theta_perturbation_for_testing[0][0], 7)
        self.assertAlmostEqual(result_theta[1][0], prefactor / theta_perturbation_for_testing[1][0], 7)
        self.assertAlmostEqual(result_theta[0][1], prefactor / theta_perturbation_for_testing[0][1], 7)
        self.assertAlmostEqual(result_theta[1][1], prefactor / theta_perturbation_for_testing[1][1], 7)
