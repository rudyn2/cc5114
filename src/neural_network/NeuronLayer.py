from src.neural_network.Neuron import Neuron
from src.neural_network.exceptions import NotAvailableActivationFunction
from src.neural_network.activation_function import Sigmoid, Step
import numpy as np


class NeuronLayer:

    def __init__(self, n_input, n_neurons, activation_func: str):

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.layer = [Neuron(n_input, self.__parse_activation_function_name(activation_func)) for _ in range(n_neurons)]

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
        try:
            assert input.shape[1] == self.n_input, "The dimension of the input is not valid. Expected (n, {})" \
                                                   "and got {}".format(self.n_input, input.shape)
        except IndexError:
            assert input.shape[0] == self.n_input, "The dimension of the input is not valid. Expected (n, {})" \
                                                   "and got {}".format(self.n_input, input.shape)

        output = [neuron.feed(input) for neuron in self.layer]
        return output

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



