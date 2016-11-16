
# In this file, we test some tuple operations.


import unittest

from tuple_operations import tuple_sum, tuple_difference, tuple_inner_product, tuple_multiply
from tuple_operations import vector_sum, vector_difference, vector_inner_product, vector_multiply


tuple1_for_testing = (3, 5)
tuple2_for_testing = (5, 7)


class TestTupleOperations(unittest.TestCase):

    def test_tuple_sum(self):
        tuple1_plus_tuple2 = tuple_sum(tuple1_for_testing, tuple2_for_testing)
        self.assertEqual((8, 12), tuple1_plus_tuple2)

    def test_tuple_difference(self):
        tuple2_minus_tuple1 = tuple_difference(tuple2_for_testing, tuple1_for_testing)
        self.assertEqual((2, 2), tuple2_minus_tuple1)

    def test_tuple_inner_product(self):
        tuple1_inner_product_tuple2 = tuple_inner_product(tuple2_for_testing, tuple1_for_testing)
        expected_inner_product = 50
        self.assertEqual(tuple1_inner_product_tuple2, expected_inner_product)

    def test_tuple_multiply(self):
        tuple1_times_2 = tuple_multiply(2, tuple1_for_testing)
        expected_tuple = (6, 10)
        self.assertEqual(tuple1_times_2, expected_tuple)


tuple3_for_testing = (7, 9)
tuple4_for_testing = (9, 11)

vector1_for_testing = [tuple1_for_testing, tuple2_for_testing]
vector2_for_testing = [tuple3_for_testing, tuple4_for_testing]


class TestVectorOperations(unittest.TestCase):

    def test_vector_sum(self):
        vector1_plus_vector2 = vector_sum(vector1_for_testing, vector2_for_testing)
        self.assertEqual([(10, 14), (14, 18)], vector1_plus_vector2)

    def test_vector_difference(self):
        vector2_minus_vector1 = vector_difference(vector2_for_testing, vector1_for_testing)
        self.assertEqual([(4, 4), (4, 4)], vector2_minus_vector1)

    def test_vector_inner_product(self):
        vector1_inner_product_vector2 = vector_inner_product(vector2_for_testing, vector1_for_testing)
        expected_inner_product = 21 + 45 + 45 + 77
        self.assertEqual(vector1_inner_product_vector2, expected_inner_product)

    def test_vector_multiply(self):
        vector1_times_2 = vector_multiply(2, vector1_for_testing)
        expected_vector = [(6, 10), (10, 14)]
        self.assertEqual(vector1_times_2, expected_vector)
