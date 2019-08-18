from src.neural_network.activation_function import Sigmoid, ActivationFunc
import numpy as np


class Neuron:

    def __init__(self, n_input: int, activation_func: ActivationFunc):
        self.n_input = n_input
        self.activation_func = activation_func
        self.w = None
        self.bias = None
        self.last_value = None

    def feed(self, x: np.ndarray) -> float:
        """
        This methods makes the neuron fire using the activation function.
        :param x:                   A numpy array 1D of same length than n input.
        :return:                    A float value.
        """
        assert self.w is not None, "This neuron doesn't has weights."
        try:
            assert x.shape[1] == self.n_input, "The input shape is not allowed."
        except IndexError:
            assert x.shape[0] == self.n_input, "The input shape is not allowed."

        pre_compute = np.matmul(x, self.w) + self.bias
        result = self.activation_func.get_value(pre_compute)
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

