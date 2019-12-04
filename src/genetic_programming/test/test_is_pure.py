from arboles import *
from abstract_syntax_tree import AST
import unittest
import random

random.seed(41)

allowed_functions = [AddNode, SubNode]
allowed_terminals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ast_generator = AST(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)

sample_tree_1 = ast_generator(max_depth=3)     # this has repeated terminals
sample_tree_2 = ast_generator(max_depth=2)     # this doesn't have repeated terminals


class TestIsPureMethod(unittest.TestCase):

    def test_false_pure(self):
        self.assertFalse(sample_tree_1.is_pure(feed_dict={'values': []}))

    def test_true_pure(self):
        self.assertTrue(sample_tree_2.is_pure(feed_dict={'values': []}))


class TestGetDepth(unittest.TestCase):

    def test_length(self):
        self.assertEqual(sample_tree_1.get_depth(), 3)
        self.assertEqual(sample_tree_2.get_depth(), 2)


if __name__ == '__main__':
    unittest.main()

