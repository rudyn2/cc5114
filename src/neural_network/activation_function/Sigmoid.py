from src.neural_network.activation_function.ActivationFunc import ActivationFunc
import numpy as np


class Sigmoid(ActivationFunc):

    def get_value(self, x):
        return 1/(1 + np.exp(x))

    def get_derivative(self, x):
        return self.get_value(x)*(1-self.get_value(x))
