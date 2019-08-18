import unittest
import numpy as np
from src.neural_network.NeuronLayer import NeuronLayer


class NeuronLayerTest(unittest.TestCase):

    def setUp(self):
        self.neuron_layer = NeuronLayer(3, 2, 'sigmoid')


class UpdateLayerTest(NeuronLayerTest):

    def runTest(self):
        """
        Test the update of the layer.
        """
        weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        bias = np.array([0.5, 0.2])
        self.neuron_layer.update(weights, bias)
        neuron_1_weights = sum(self.neuron_layer.layer[0].w == weights[0, :])
        neuron_2_weights = sum(self.neuron_layer.layer[1].w == weights[1, :])
        self.assertEqual(neuron_1_weights, 3)
        self.assertEqual(neuron_2_weights, 3)


class FeedLayerTest(NeuronLayerTest):

    def runTest(self):
        """
        Test the feeding of the layer.
        """

        weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        bias = np.array([0.5, 0.2])
        self.neuron_layer.update(weights, bias)
        output = self.neuron_layer.feed(np.array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(0.1824255238063, output[0])
        self.assertAlmostEqual(0.23147521650098, output[1])



