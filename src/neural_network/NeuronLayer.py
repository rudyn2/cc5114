from src.neural_network.Neuron import Neuron
from src.neural_network.exceptions import NotAvailableActivationFunction
from src.neural_network.activation_function import Sigmoid, Step
import numpy as np


class NeuronLayer:

    def __init__(self, n_input, n_neurons, activation_func: str):

        # Each neuron of the layer has n inputs
        self.n_input = n_input

        # Number of neurons in the layer
        self.n_neurons = n_neurons

        # Builds the layer
        self.layer = [Neuron(n_input, self.__parse_activation_function_name(activation_func)) for _ in range(n_neurons)]
        self.last_values = None

    @staticmethod
    def __parse_activation_function_name(name: str):

        allowed = ['sigmoid', 'step']
        if name.lower() not in allowed:
            raise NotAvailableActivationFunction.NotAvailableActivationFunction

        parser = {
            'sigmoid': Sigmoid.Sigmoid(),
            'step': Step.Step()
        }
        return parser[name.lower()]

    def feed(self, input):
        """
        It receives the input data and outputs an array with the result per neuron.
        :param input:                       A Numpy array with the input data.
        :return:                            A Numpy array with the output.
        """
        assert input.shape[0] == self.n_input, "The dimension of the input is not valid. Expected (n, {}) " \
                                                   "and got {}".format(self.n_input, input.shape)

        output = [neuron.feed(input) for neuron in self.layer]
        self.last_values = np.array(output)
        return self.last_values

    def update(self, weights: np.ndarray, bias: np.ndarray):
        """
        It receives a matrix with the weights and an array with the bias for the layer.
        :param weights:                     A 2D Numpy Array.
        :param bias:                        A 1D Numpy Array.
        """
        assert weights.shape[0] == len(self.layer), "You must provide weights for all the neurons."
        assert weights.shape[1] == self.n_input, "The dimension of the provided weights is not valid. " \
                                                 "Expected ({}, {}) and got {}".format(len(self.layer), self.n_input, weights.shape)

        assert len(bias) == weights.shape[0], "The shape of bias is not equals to the number of rows of the weights."

        for index in range(weights.shape[0]):
            neuron_weight = weights[index, :]
            neuron_bias = bias[index]
            self.layer[index].update(neuron_weight, neuron_bias)

    def random_init(self):
        """
        It updates each neuron with random weights and bias.
        """
        weights = np.random.random(size=(len(self.layer), self.n_input))
        # weights = np.array([0.1581, 0.8045]).T
        # weights = np.ones(shape=(len(self.layer), self.n_input))
        bias = np.random.random(size=(weights.shape[0], 1))
        # bias = np.array([0.8729])
        self.update(weights, bias)


