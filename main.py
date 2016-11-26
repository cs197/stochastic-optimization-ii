from data_from_ng import REVIEWS, USERS, MOVIES

from optimization_objective import make_optimization_objective

from stochastic_perturbation import generate_perturbation

from simultaneous_perturbation import simultaneous_perturbation, gain_a, gain_c

# The following could have been eliminated if I had wanted to add a dependency on numpy.
from tuple_operations import vector_sum, vector_multiply, tuple_inner_product

from random import seed


# This is an implementation of Equations 6.4 and 6.6 of Spall:
# http://www.jhuapl.edu/spsa/PDF-SPSA/Handbook04_StochasticOptimization.pdf
# Equation 6.6 is bizarre. Be sure to understand it before adopting this algorithm.
# Spall's k <=> my idx.
# Spall's a_k <=> my gain_a(idx).
# Spall's c_k <=> my gain_c(idx).
# I consider it an error, albeit an instructive one, that I am using a general-purpose
# optimization algorithm to perform this particular optimization. The collaborative-filtering
# optimization function is complicated looking but actually just a quadratic, and is
# therefore amenable to special-purpose techniques.
def optimize_ng_example():
    lamb = 0.1  # I'd call this lambda as Ng does, except in Python lambda is a keyword.
    optimization_objective_function = make_optimization_objective(lamb, REVIEWS)

    convergence_criterion = 0.00000001

    # Our initial guess -- totally random of course.
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
            print "Terminating after {0} iterations. The optimization value is {1}.".format(str(idx),
                                                                                            str(objective_value))
            break
        else:
            if idx % 100 == 0:
                print "After {0} iterations, the optimization value is {1}.".format(str(idx),
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
        print "{0}'s rating for \"{1}\": predicted {2}, actual {3}.".format(USERS[idx_user], MOVIES[idx_movie],
                                                                            prediction, rating)


if __name__ == "__main__":
    seed("Let's make this reproducible, eh?")
    optimize_ng_example()
