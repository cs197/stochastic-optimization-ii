from data_from_ng import REVIEWS

from optimization_objective import make_optimization_objective

from stochastic_perturbation import generate_perturbation

from simultaneous_perturbation import simultaneous_perturbation

from tuple_operations import vector_sum, vector_multiply

from random import seed

from math import exp, log

# Spall's gain parameters

gain_a_multiplier = 0.01
gain_a_fudge = 2.0
alpha = 0.2


def gain_a(idx):
    return gain_a_multiplier / exp(alpha * log(idx + 1 + gain_a_fudge))


gain_c_multiplier = 0.01
gamma = 0.2


def gain_c(idx):
    return gain_c_multiplier / exp(gamma * log(idx + 1))


# This is an implementation of Equation 6.4 of Spall
# in http://www.jhuapl.edu/spsa/PDF-SPSA/Handbook04_StochasticOptimization.pdf
# Spall does not give a lot of guidance on how the gains a_k and c_k should be chosen, and the choices affect
# the convergence.  My routine appears to converge with my choices, but I am not confident it is correct.
def optimize_ng_example():
    lamb = 1.0  # I'd call this lambda, except in Python lambda is a keyword. Initialize to 1.0. What should it be?
    optimization_objective_function = make_optimization_objective(lamb, REVIEWS)

    convergence_criterion = 0.0001

    # Our initial guess -- totally random -- no method or logic here.
    idx = 1
    c = gain_c(idx)
    x_list, theta_list = generate_perturbation(c)
    objective_value = optimization_objective_function(x_list, theta_list)

    idx = 1
    while True:
        objective_old_value = objective_value
        c = gain_c(idx)
        x_perturbation, theta_perturbation = generate_perturbation(c)
        gradient_x, gradient_theta = simultaneous_perturbation(optimization_objective_function,
                                                               x_list, theta_list,
                                                               x_perturbation, theta_perturbation)

        a = gain_a(idx)
        delta_x = vector_multiply(-1.0 * a, gradient_x)
        delta_theta = vector_multiply(-1.0 * a, gradient_theta)

        x_list = vector_sum(x_list, delta_x)
        theta_list = vector_sum(theta_list, delta_theta)
        objective_value = optimization_objective_function(x_list, theta_list)
        if abs(objective_value - objective_old_value) < convergence_criterion:
            print "Terminating after {0} iterations. The optimization value is {1}".format(str(idx),
                                                                                           str(objective_value))
            break
        else:
            if idx % 10 == 0:
                print "After {0} iterations, the optimization value is {1}".format(str(idx),
                                                                                   str(objective_value))

            idx += 1

    print "Feature vector for the movies is: " + str(x_list) + "."
    print "Feature affinity vector for the users is: " + str(theta_list) + "."


if __name__ == "__main__":
    seed("Let's make this reproducible, eh?")
    optimize_ng_example()
