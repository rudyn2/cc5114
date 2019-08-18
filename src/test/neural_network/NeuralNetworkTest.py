import unittest
from src.neural_network.NeuralNetwork import NeuralNetwork


class NeuralNetworkTest(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork([3, 2, 1], 'sigmoid')


class BuildNeuralNetworkTest(NeuralNetworkTest):

    def runTest(self):

        self.assertEqual(self.nn.n_input, 3)
        self.assertEqual(self.nn.neural_network[0].n_neurons, 2)
        self.assertEqual(self.nn.neural_network[1].n_neurons, 1)

