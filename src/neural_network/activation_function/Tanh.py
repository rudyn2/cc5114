from src.neural_network.activation_function.ActivationFunc import ActivationFunc
import numpy as np


class Tanh(ActivationFunc):

    def get_value(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def get_derivative(self, x):
        return 1-self.get_value(x)**2
