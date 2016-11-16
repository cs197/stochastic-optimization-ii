
from tuple_operations import tuple_inner_product, vector_inner_product

# After several motivational lectures, in a lecture titled "Collaborative Filtering Algorithm," at 1:10 Ng reveals
# his optimization objective for collaborative filtering. However, he then simplifies it by having x and theta be
# n-dimensional vectors (where n is the number of features), rather than n+1 dimensional vectors. This leads to a
# simplification discussed in the video about 5:45.

# In this file, we implement optimization_objective. (Ng calls it J.)


# As is so often the case in these optimization problems, I will make a function that returns optimization_objective,
# rather than defining optimization_objective directly. This is my way of encapsulating that the function is best
# defined with reviews in its enclosing scope. Globals are messy. Long argument lists are also messy.
# Closures allow us to avoid either of those messes.  Note there is very little cost
# to passing reviews around even if it is a very long list. Its semantic is pass-by-reference, not pass-by-copy.
def make_optimization_objective(lamb, reviews):

    def optimization_objective(x_list, theta_list):
        # As you can see at 1:10 of the video, what Ng calls J is decently complex.
        # Thanks to the various vector and tuple operations defined in tuple_operations.py, this
        # complexity is somewhat hidden.

        first_sum = 0.0
        second_sum = 0.0
        third_sum = 0.0

        for review in reviews:
            i = review.movie
            j = review.user
            y_i_j = review.rating

            x_i = x_list[i]
            theta_j = theta_list[j]

            delta = tuple_inner_product(theta_j, x_i) - y_i_j

            first_sum += delta * delta

        second_sum += vector_inner_product(x_list, x_list)

        third_sum += vector_inner_product(theta_list, theta_list)

        return (first_sum + lamb * second_sum + lamb * third_sum) / 2.0

    return optimization_objective
