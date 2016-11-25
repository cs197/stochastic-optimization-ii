from data_from_ng import REVIEWS, USERS, MOVIES

from optimization_objective import make_optimization_objective

from stochastic_perturbation import generate_perturbation

from simultaneous_perturbation import simultaneous_perturbation, gain_a, gain_c

from tuple_operations import vector_sum, vector_multiply, tuple_inner_product

from random import seed


# This is an implementation of Equation 6.4 of Spall
# in http://www.jhuapl.edu/spsa/PDF-SPSA/Handbook04_StochasticOptimization.pdf
# the convergence.  My routine appears to converge with my choices, but I am not confident it is correct.
def optimize_ng_example():
    lamb = 0.1  # I'd call this lambda, except in Python lambda is a keyword.
    optimization_objective_function = make_optimization_objective(lamb, REVIEWS)

    convergence_criterion = 0.00000001

    # Our initial guess -- totally random -- no method or logic here.
    idx = 0
    a = gain_a(idx)
    x_list, theta_list = generate_perturbation(a)
    objective_value = optimization_objective_function(x_list, theta_list)

    while True:
        idx += 1
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

    print "Feature vector for the movies is: " + str(x_list) + "."
    print "Feature affinity vector for the users is: " + str(theta_list) + "."

    for review in REVIEWS:
        idx_user = review.user
        idx_movie = review.movie
        features_affinity = theta_list[idx_user]
        features = x_list[idx_movie]
        prediction = tuple_inner_product(features_affinity, features)
        rating = review.rating
        print "{0}'s rating for \"{1}\": predicted {2}, actual {3}".format(USERS[idx_user], MOVIES[idx_movie],
                                                                           prediction, rating)


if __name__ == "__main__":
    seed("Let's make this reproducible, eh?")
    optimize_ng_example()
