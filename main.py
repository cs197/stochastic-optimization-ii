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


# In simultaneous_perturbation.py, you can implement Spall's algorithm.
#
# The following is a dummy implementation!! It is your job to implement it.
# Really, what this amounts to is implementing Equation 6.4 of Spall.
# http://www.jhuapl.edu/spsa/PDF-SPSA/Handbook04_StochasticOptimization.pdf
# A tricky thing that I don't have much sense of is how the gain vectors that Spall calls a_k and c_k should
# be chosen.
def optimize_ng_example():
    lamb = 1.0  # I'd call this lambda, except in Python lambda is a keyword. Initialize to 1.0. What should it be?
    optimization_objective_function = make_optimization_objective(lamb, REVIEWS)

    convergence_criterion = 0.0001

    # Our initial guess -- totally random -- no method or logic here.
    idx = 1
    c = gain_c(idx)
    x_list, theta_list = generate_perturbation(c)
    optimization_objective_value = optimization_objective_function(x_list, theta_list)

    print "Initially, the optimization objective is " + str(optimization_objective_value)


if __name__ == "__main__":
    seed("Let's make this reproducible, eh?")
    optimize_ng_example()
