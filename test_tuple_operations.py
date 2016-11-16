
# In this file, we test some tuple operations.


import unittest

from tuple_operations import tuple_sum, tuple_difference, tuple_inner_product, tuple_multiply


tuple1_for_testing = (3, 5)
tuple2_for_testing = (5, 7)


class TestTupleOperations(unittest.TestCase):

    def test_tuple_sum(self):
        vector1_plus_vector2 = tuple_sum(tuple1_for_testing, tuple2_for_testing)
        self.assertEqual((8, 12), vector1_plus_vector2)

    def test_tuple_difference(self):
        vector1_minus_vector2 = tuple_difference(tuple2_for_testing, tuple1_for_testing)
        self.assertEqual((2, 2), vector1_minus_vector2)

    def test_tuple_inner_product(self):
        tuple1_inner_product_tuple2 = tuple_inner_product(tuple2_for_testing, tuple1_for_testing)
        expected_inner_product = 50
        self.assertEqual(tuple1_inner_product_tuple2, expected_inner_product)

    def test_tuple_multiply(self):
        tuple1_times_2 = tuple_multiply(2, tuple1_for_testing)
        expected_tuple = (6, 10)
        self.assertEqual(tuple1_times_2, expected_tuple)
