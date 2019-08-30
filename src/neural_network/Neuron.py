from src.neural_network.activation_function import Sigmoid, Step, Tanh, ActivationFunc
from src.neural_network.exceptions.NotAvailableActivationFunction import NotAvailableActivationFunction
import numpy as np


class Neuron:
    """
    Neuron class. Minimal entity of a neural network.
    """

    def __init__(self, n_input: int, activation_func: str):
        """
        Constructor for the Neuron class.
        :param n_input:                             Number of inputs of the neuron (not considering the bias).
        :param activation_func:                     Name of the activation function asigned.
        """
        self.n_input = n_input
        self.activation_func = self.__parse_activation_function_name(activation_func)
        self.w = None
        self.bias = None
        self.last_value = None

    @staticmethod
    def __parse_activation_function_name(name: str) -> ActivationFunc:
        """
        This methods receives the name of a activation function and returns the respective activation function
        object.
        :param name:                                String. Name of the activation function (sigmoid, tanh, step).
        :return:                                    Activation function object.
        """

        # Defines the allowed activation function. HARDCODED: THIS HAS TO BE MANUALLY UPDATED.
        allowed = ['sigmoid', 'step', 'tanh']
        if name.lower() not in allowed:
            raise NotAvailableActivationFunction

        parser = {
            'sigmoid': Sigmoid.Sigmoid(),
            'step': Step.Step(),
            'tanh': Tanh.Tanh()
        }
        return parser[name.lower()]

    def feed(self, x: np.ndarray) -> float:
        """
        This methods makes the neuron fire using the activation function.
        :param x:                   A numpy array 1D of same length than n input with shape (n, 1).
        :return:                    A float value.
        """
        assert self.w is not None, "This neuron doesn't has weights."
        assert x.shape[0] == self.n_input, "The input shape is not allowed."

        pre_compute = np.matmul(x.T, self.w) + self.bias
        result = self.activation_func.get_value(pre_compute[0])
        self.last_value = result
        return result

    def update(self, w: np.ndarray, bias: float):
        """
        Updates the weights and bias of the neuron.
        :param w:                   A numpy array with the weights.
        :param bias:                A float value with the bias.
        :return:                    A float value.
        """
        assert len(w) == self.n_input, "The shape of the weights must be {}".format(self.n_input)
        self.w = w
        self.bias = bias

