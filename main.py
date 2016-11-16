from data_from_ng import REVIEWS

from optimization_objective import make_optimization_objective

from stochastic_perturbation import generate_perturbation

from simultaneous_perturbation import simultaneous_perturbation

from tuple_operations import vector_sum, vector_multiply

from random import seed


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

    gain_a = 1.0  # One of Spall's gain factors. Should be decreased as the optimization converges.
    gain_c = 1.0  # The other of Spall's gain factors. Also should be decreased as the optimization converges.
    convergence_criterion = 0.0001

    # Our initial guess -- totally random -- no method or logic here.
    x_list, theta_list = generate_perturbation(gain_a)
    optimization_objective_value = optimization_objective_function(x_list, theta_list)

    print "Initially, the optimization objective is " + str(optimization_objective_value)


if __name__ == "__main__":
    seed("Let's make this reproducible, eh?")
    optimize_ng_example()
