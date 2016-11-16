
# First we implement some basic mathematical operations on tuples.


def tuple_inner_product(v1, v2):

    assert len(v1) == len(v2)

    result = 0.0

    for idx in range(len(v1)):
        result += v1[idx] * v2[idx]

    return result


def tuple_sum(v, v_perturbation):
    assert len(v) == len(v_perturbation)

    return tuple([x + x_perturbation for x, x_perturbation in zip(v, v_perturbation)])


def tuple_difference(v, v_perturbation):
    assert len(v) == len(v_perturbation)

    return tuple([x - x_perturbation for x, x_perturbation in zip(v, v_perturbation)])


def tuple_multiply(scalar, a_tuple):

    return tuple([scalar * a_value for a_value in a_tuple])


# Wow, we also need operations on vectors of tuples. This is getting pretty dense.

def vector_inner_product(list_a, list_b):

    assert len(list_a) == len(list_b)

    result = 0.0

    for idx in range(len(list_a)):
        tuple_a = list_a[idx]
        tuple_b = list_b[idx]
        result += tuple_inner_product(tuple_a, tuple_b)

    return result


def vector_sum(x, x_perturbation):
    assert len(x) == len(x_perturbation)

    return [tuple_sum(x_tuple, x_tuple_perturbation) for x_tuple, x_tuple_perturbation in zip(x, x_perturbation)]


def vector_difference(x, x_perturbation):
    assert len(x) == len(x_perturbation)

    return [tuple_difference(x_tuple, x_tuple_perturbation) for x_tuple, x_tuple_perturbation in zip(x, x_perturbation)]


def vector_multiply(scalar, a_vector):

    return [tuple_multiply(scalar, a_tuple) for a_tuple in a_vector]
