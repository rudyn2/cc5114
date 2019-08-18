import unittest
import numpy as np
from src.neural_network.Neuron import Neuron
from src.neural_network.activation_function.Sigmoid import Sigmoid


class NeuronTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.neuron = Neuron(5, Sigmoid())


class UpdateNeuronTest(NeuronTest):

    def runTest(self):
        """
        Test the update of the neuron.
        """
        weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        bias = 0.5
        self.neuron.update(weights, bias)
        truth = sum(self.neuron.w == weights)
        self.assertEqual(truth, 5)
        self.assertEqual(self.neuron.bias, bias)


class FeedNeuronTest(NeuronTest):

    def runTest(self):
        """
        Test the feeding of the neuron.
        """
        weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        bias = 0.5
        self.neuron.update(weights, bias)
        output = self.neuron.feed(np.array([0, 0, 0, 0, 1.0]))
        self.assertAlmostEqual(output, 0.182425523806)

