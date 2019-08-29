from src.neural_network.activation_function.ActivationFunc import ActivationFunc
import numpy as np


class Tanh(ActivationFunc):
    """
    Tanh activation function class.
    """

    def get_value(self, x):
        """
        Maps from x to f(x) where f(x) corresponds to the tanh function.
        :param x:                           Input of function.
        :return:                            Mapped value.
        """
        a = np.exp(x)
        b = np.exp(-x)
        return (a-b)/(a+b)

    def get_derivative(self, x):
        """
        Maps from x to f(x) where f(x) corresponds to the derivative of tanh function.
        :param x:                           Input of function.
        :return:                            Mapped value.
        """
        return 1-self.get_value(x)**2
