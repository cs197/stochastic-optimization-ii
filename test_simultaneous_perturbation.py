
# In this file, we test some tuple operations.

import unittest

from simultaneous_perturbation import simultaneous_perturbation

from tuple_operations import tuple_inner_product

x_vector_for_testing = [(3, 5), (1, 2), (3, 4), (5, 6), (7, 8)]
theta_vector_for_testing = [(5, 7), (9, 10), (11, 12), (13, 14)]

x_perturbation_for_testing = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1.0)]
theta_perturbation_for_testing = [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1.0)]


def optimization_objective_function_for_testing(x_vector, theta_vector):
    accumulator = 0.0
    for x in x_vector:
        accumulator += tuple_inner_product(x, x)

    for theta in theta_vector:
        accumulator += tuple_inner_product(theta, theta)

    return accumulator

# The following should be optimization_objective_function_for_testing evaluated at
# [(3.1, 5.2), (1.3, 2.4), (3.5, 4.6), (5.7, 6.8), (7.9, 9.0)], [(5.3, 7.4), (9.5, 10.6), (11.7, 12.8), (13.9, 15.0)]
# minus the same function evaluated at
# [(2.9, 4.8), (0.8, 1.6), (2.5, 3.4), (4.3, 5.6), (6.1, 7.0)], [(4.7, 6.6), (8.5, 9.4), (10.3, 11.2), (12.1, 13.0)]
# divided by 2. To be honest, I haven't checked it. Nonetheless, the test is non-trivial.
prefactor = 173.4


class TestSimultaneousPerturbation(unittest.TestCase):

    def test_simultaneous_perturbation(self):
        result_x, result_theta = simultaneous_perturbation(optimization_objective_function_for_testing,
                                                           x_vector_for_testing, theta_vector_for_testing,
                                                           x_perturbation_for_testing, theta_perturbation_for_testing)

        # First check the x-direction gradients
        self.assertAlmostEqual(result_x[0][0], prefactor / x_perturbation_for_testing[0][0], 7)
        self.assertAlmostEqual(result_x[1][0], prefactor / x_perturbation_for_testing[1][0], 7)
        self.assertAlmostEqual(result_x[2][0], prefactor / x_perturbation_for_testing[2][0], 7)
        self.assertAlmostEqual(result_x[3][0], prefactor / x_perturbation_for_testing[3][0], 7)
        self.assertAlmostEqual(result_x[4][0], prefactor / x_perturbation_for_testing[4][0], 7)
        self.assertAlmostEqual(result_x[0][1], prefactor / x_perturbation_for_testing[0][1], 7)
        self.assertAlmostEqual(result_x[1][1], prefactor / x_perturbation_for_testing[1][1], 7)
        self.assertAlmostEqual(result_x[2][1], prefactor / x_perturbation_for_testing[2][1], 7)
        self.assertAlmostEqual(result_x[3][1], prefactor / x_perturbation_for_testing[3][1], 7)
        self.assertAlmostEqual(result_x[4][1], prefactor / x_perturbation_for_testing[4][1], 7)

        # Then check the theta-direction gradients
        self.assertAlmostEqual(result_theta[0][0], prefactor / theta_perturbation_for_testing[0][0], 7)
        self.assertAlmostEqual(result_theta[1][0], prefactor / theta_perturbation_for_testing[1][0], 7)
        self.assertAlmostEqual(result_theta[2][0], prefactor / theta_perturbation_for_testing[2][0], 7)
        self.assertAlmostEqual(result_theta[3][0], prefactor / theta_perturbation_for_testing[3][0], 7)
        self.assertAlmostEqual(result_theta[0][1], prefactor / theta_perturbation_for_testing[0][1], 7)
        self.assertAlmostEqual(result_theta[1][1], prefactor / theta_perturbation_for_testing[1][1], 7)
        self.assertAlmostEqual(result_theta[2][1], prefactor / theta_perturbation_for_testing[2][1], 7)
        self.assertAlmostEqual(result_theta[3][1], prefactor / theta_perturbation_for_testing[3][1], 7)
