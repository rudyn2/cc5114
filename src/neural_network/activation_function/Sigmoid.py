from src.neural_network.activation_function.ActivationFunc import ActivationFunc
import numpy as np


class Sigmoid(ActivationFunc):
    """
    Sigmoid activation function class.
    """

    def get_value(self, x):
        """
        Maps from x to f(x) where f(x) corresponds to the sigmoid function.
        :param x:                           Input of function.
        :return:                            Mapped value.
        """
        return 1/(1 + np.exp(-x))

    def get_derivative(self, x):
        """
        Maps from x to f(x) where f(x) corresponds to the derivative of sigmoid function.
        :param x:                           Input of function.
        :return:                            Mapped value.
        """
        return self.get_value(x)*(1-self.get_value(x))
